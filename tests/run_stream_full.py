"""Build and run exact-input STREAM source/MLIR pairs in the same Cyclotron config.

The handwritten source and input formulas are read from radiance-kernels. All
generated files stay in --out, so a dirty FPGA/kernel checkout is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from generate_stream_full import generate
from generate_stream import KINDS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(argv: list[str], *, cwd: Path, log: Path) -> str:
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    with log.open("w") as output:
        process = subprocess.run(argv, cwd=cwd, env=env, stdout=output,
                                 stderr=subprocess.STDOUT, text=True)
    result = log.read_text()
    if process.returncode:
        raise RuntimeError(f"{' '.join(argv[:2])} failed; see {log}\n{result[-2000:]}")
    return result


def symbol(nm: Path, elf: Path, name: str) -> str:
    listing = subprocess.run([str(nm), "-n", str(elf)], capture_output=True,
                             text=True, check=True).stdout
    match = re.search(rf"^([0-9a-fA-F]+) [A-Za-z] {name}$", listing, re.M)
    if not match:
        raise RuntimeError(f"{elf} has no {name} symbol")
    return f"0x{match.group(1)}"


def prepare_data(source: Path, kind: str, elements: int, out: Path) -> dict:
    spec = importlib.util.spec_from_file_location("radiance_stream_source", source / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.OUTPUT[kind]
    inputs = {}
    for name in "abc":
        if name != output:
            path = out / f"{name}.bin"
            module.write_input(path, name, elements)
            inputs[name] = digest(path)
    assembly = ['.section .data,"aw",@progbits']
    for name in "abc":
        if name != output:
            assembly += [".balign 64", f".globl stream_{name}", f"stream_{name}:",
                         f'.incbin "{name}.bin"']
    assembly += ['.section .bss,"aw",@nobits', '.balign 64',
                 '.globl stream_guard_before', 'stream_guard_before:', '.zero 64',
                 f'.globl stream_{output}', f'stream_{output}:', f'.zero {elements * 4}',
                 '.globl stream_guard_after', 'stream_guard_after:', '.zero 64']
    (out / "data.S").write_text("\n".join(assembly) + "\n")
    (out / "generated").mkdir(exist_ok=True)
    samples = min(64, elements) if elements > 1024 else 0
    positions = ([i * (elements - 1) // (samples - 1) for i in range(samples)]
                 if samples else range(elements))
    expected = module.digest(module.float_word(module.result_value(kind, i))
                             for i in positions)
    (out / "generated/config.h").write_text(
        f"#pragma once\n#define STREAM_KIND {module.KINDS[kind]}\n"
        f"#define STREAM_ELEMENTS {elements}u\n"
        f"#define STREAM_READBACK_SAMPLES {samples}u\n"
        f"#define STREAM_EXPECTED_READBACK_DIGEST 0x{expected:016x}ULL\n")
    return {"input_sha256": inputs, "data_assembly_sha256": digest(out / "data.S"),
            "source_run_sha256": digest(source / "run.py"),
            "expected_host_readback_digest": f"{expected:016x}",
            "host_readback_samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=(*KINDS, "all"), default="all")
    parser.add_argument("--elements", type=int, default=1_048_576)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="radiance-kernels/kernels/stream")
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--muon-clang", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-include", type=Path, required=True)
    parser.add_argument("--libcxx-include", type=Path, required=True)
    parser.add_argument("--config-site-dir", type=Path, required=True)
    parser.add_argument("--linker-script", type=Path, required=True)
    parser.add_argument("--tohost", type=Path, required=True)
    parser.add_argument("--cyclotron-check", type=Path, required=True)
    parser.add_argument("--cyclotron-config", type=Path, required=True)
    args = parser.parse_args()
    for name in ("source", "muon_opt", "llvm_bin", "muon_clang", "runtime_archive",
                 "runtime_include", "libcxx_include", "config_site_dir",
                 "linker_script", "tohost", "cyclotron_check", "cyclotron_config"):
        setattr(args, name, getattr(args, name).resolve())
    if args.elements < 2 or args.elements & 1:
        parser.error("--elements must be an even integer at least two")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    llvm_bin = args.llvm_bin.resolve()
    clang = str(args.muon_clang.resolve())
    muon_opt = args.muon_opt.resolve()
    include = out / "include"
    include.mkdir(exist_ok=True)
    assertion_handler = args.libcxx_include.resolve().parent / "vendor/llvm/default_assertion_handler.in"
    (include / "__assertion_handler").write_bytes(assertion_handler.read_bytes())
    common = [clang, "-target", "riscv32-unknown-elf", "-march=rv32im_zfinx_zhinx",
              "-mabi=ilp32", "-Xclang", "-target-feature", "-Xclang", "+vortex",
              "-O3", "-mcmodel=medany", "-fno-rtti", "-fno-exceptions",
              "-fdata-sections", "-ffunction-sections", "-I", str(args.runtime_include),
              "-I", str(out), "-isystem", str(include),
              "-isystem", str(args.libcxx_include),
              "-isystem", str(args.config_site_dir),
              "-DRADIANCE", "-DRADIANCE_DEVICE", "-DNDEBUG", "-DLLVM_VORTEX"]
    records = []
    for kind in (KINDS if args.kind == "all" else (args.kind,)):
        case = out / kind
        case.mkdir(exist_ok=True)
        data = prepare_data(source, kind, args.elements, case)
        shutil.copy2(source / "kernel.cpp", case / "kernel.cpp")
        mlir = case / "kernel.mlir"
        mlir.write_text(generate(kind, args.elements))
        run([str(muon_opt), "--lower-muon-runtime", str(mlir), "-o",
             str(case / "kernel.lowered.mlir")], cwd=case, log=case / "lower.log")
        run([str(llvm_bin / "mlir-opt"), str(case / "kernel.lowered.mlir"),
             "--convert-scf-to-cf", "--convert-arith-to-llvm", "--convert-func-to-llvm",
             "--convert-cf-to-llvm", "--reconcile-unrealized-casts", "-o",
             str(case / "kernel.llvm.mlir")], cwd=case, log=case / "convert.log")
        ir = case / "kernel.ll"
        run([str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
             str(case / "kernel.llvm.mlir"), "-o", str(ir)], cwd=case,
            log=case / "translate.log")
        ir.write_text(ir.read_text().replace("getelementptr inbounds nuw ",
                                             "getelementptr inbounds "))
        (case / "main.c").write_text(
            "extern void entry(void);\nunsigned __mu_num_warps = 4;\n"
            "int main(void) { entry(); return 0; }\n")
        runtime_cpp = Path(__file__).resolve().parents[1] / "runtime/muon_mlir_runtime.cpp"
        objects = {
            "data": ["-x", "assembler-with-cpp", "-c", "data.S"],
            "source": ["-x", "c++", "-std=c++20", "-c", "kernel.cpp"],
            "mlir": ["-x", "ir", "-c", "kernel.ll"],
            "main": ["-x", "c", "-c", "main.c"],
            "runtime": ["-x", "c++", "-std=c++20", "-c", str(runtime_cpp)],
        }
        for name, flags in objects.items():
            run(common + flags + ["-o", f"{name}.o"], cwd=case,
                log=case / f"compile_{name}.log")
        case_record = {"kind": kind, "elements": args.elements, **data,
                       "kernel_source_sha256": digest(source / "kernel.cpp"),
                       "mlir_sha256": digest(mlir), "llvm_ir_sha256": digest(ir),
                       "runs": {}}
        for variant, objects_to_link in (
            ("source", ["source.o", "data.o"]),
            ("mlir", ["mlir.o", "main.o", "runtime.o", "data.o"]),
        ):
            elf = case / f"{variant}.radiance.elf"
            run(common + ["-nostdlib", "-nodefaultlibs", "-nostartfiles", "-fuse-ld=lld",
                 f"-Wl,-Bstatic,-T,{args.linker_script},-z,norelro", *objects_to_link,
                 str(args.runtime_archive), str(args.tohost), "-o", str(elf)],
                cwd=case, log=case / f"link_{variant}.log")
            output = {"copy": "c", "scale": "b", "add": "c", "triad": "a"}[kind]
            addr = symbol(llvm_bin / "llvm-nm", elf, f"stream_{output}")
            sim_log = case / f"{variant}.cyclotron.log"
            result = run([str(args.cyclotron_check), str(args.cyclotron_config), str(elf),
                          kind, addr, str(args.elements), "1" if args.timing else "0"],
                         cwd=args.cyclotron_config.parent, log=sim_log)
            match = re.search(r"simulation finished after (\d+) cycles", result)
            verified = re.search(rf"STREAM_FULL_RESULT kind={kind} elements={args.elements} "
                                 r"digest=([0-9a-f]{16})", result)
            if not match or not verified:
                raise RuntimeError(f"{variant} verification incomplete; see {sim_log}")
            case_record["runs"][variant] = {
                "elf_sha256": digest(elf), "output_address": addr,
                "output_digest": verified.group(1), "cyclotron_steps": int(match.group(1)),
            }
            print(f"{kind} {variant}: {args.elements} exact words and guards passed; "
                  f"{match.group(1)} {'timing' if args.timing else 'functional'} steps", flush=True)
        a = case_record["runs"]["source"]
        b = case_record["runs"]["mlir"]
        assert a["output_digest"] == b["output_digest"]
        if args.timing:
            case_record["slowdown_ratio"] = b["cyclotron_steps"] / a["cyclotron_steps"]
            case_record["within_10_percent"] = (
                0.90 <= case_record["slowdown_ratio"] <= 1.10)
        records.append(case_record)
        (out / "manifest.json").write_text(json.dumps({
            "schema": "muon_mlir_stream_full.v1", "timing": args.timing,
            "compiler_sha256": digest(args.muon_clang),
            "runtime_archive_sha256": digest(args.runtime_archive),
            "cyclotron_config_sha256": digest(args.cyclotron_config),
            "cyclotron_checker_sha256": digest(args.cyclotron_check),
            "cases": records}, indent=2) + "\n")
    print(out / "manifest.json")
    if args.timing and any(not case["within_10_percent"] for case in records):
        raise SystemExit("Muon MLIR exceeded the 10% STREAM timing gate; see manifest.json")


if __name__ == "__main__":
    main()
