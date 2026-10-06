"""Compare source and Muon MLIR Gather on one original Spatter JSON case."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
from pathlib import Path

from generate_spatter_full_gather import generate
from run_stream_full import digest, run, symbol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="radiance-kernels/kernels/spatter")
    parser.add_argument("--timing", action="store_true")
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
    for name in ("suite", "source", "out", "muon_opt", "llvm_bin", "muon_clang",
                 "runtime_archive", "runtime_include", "libcxx_include", "config_site_dir",
                 "linker_script", "tohost", "cyclotron_check", "cyclotron_config"):
        setattr(args, name, getattr(args, name).resolve())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    suite = json.loads(args.suite.read_text())
    if not 0 <= args.case < len(suite) or suite[args.case].get("kernel", "").lower() != "gather":
        parser.error("--case must select a Gather in the source Spatter suite")
    clone = out / "source_copy"
    clone.mkdir(exist_ok=True)
    for name in ("run.py", "plan.py", "Makefile", "kernel.cpp", "spatter_ops.hpp",
                 "host.cpp", "emit_symbols.py", "cyclotron-no-trace.patch"):
        shutil.copy2(args.source / name, clone / name)
    run(["python3", str(clone / "run.py"), "--suite", str(args.suite),
         "--case", str(args.case), "--out", str(out / "prepare"), "--prepare-only"],
        cwd=clone, log=out / "prepare.log")
    prepared = json.loads((out / "prepare/result.json").read_text())
    spec = importlib.util.spec_from_file_location("spatter_source_clone", clone / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(clone))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    case = module.normalize(suite[args.case])
    if case["collision_policy"] != "parallel":
        parser.error("this Gather route expects the source parallel policy")
    mlir = clone / "kernel.mlir"
    mlir.write_text(generate(case))
    run([str(args.muon_opt), "--lower-muon-runtime", str(mlir), "-o",
         str(clone / "kernel.lowered.mlir")], cwd=clone, log=clone / "lower.log")
    run([str(args.llvm_bin / "mlir-opt"), str(clone / "kernel.lowered.mlir"),
         "--convert-scf-to-cf", "--convert-arith-to-llvm", "--convert-func-to-llvm",
         "--convert-cf-to-llvm", "--reconcile-unrealized-casts", "-o",
         str(clone / "kernel.llvm.mlir")], cwd=clone, log=clone / "convert.log")
    ir = clone / "kernel.ll"
    run([str(args.llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
         str(clone / "kernel.llvm.mlir"), "-o", str(ir)], cwd=clone,
        log=clone / "translate.log")
    ir.write_text(ir.read_text().replace("getelementptr inbounds nuw ",
                                         "getelementptr inbounds "))
    (clone / "main.c").write_text(
        "extern void entry(void);\nunsigned __mu_num_warps = 4;\n"
        "int main(void) { entry(); return 0; }\n")
    include = out / "include"
    include.mkdir(exist_ok=True)
    assertion_handler = args.libcxx_include.parent / "vendor/llvm/default_assertion_handler.in"
    (include / "__assertion_handler").write_bytes(assertion_handler.read_bytes())
    common = [str(args.muon_clang), "-target", "riscv32-unknown-elf",
              "-march=rv32im_zfinx_zhinx", "-mabi=ilp32", "-Xclang",
              "-target-feature", "-Xclang", "+vortex", "-O3", "-mcmodel=medany",
              "-fno-rtti", "-fno-exceptions", "-fdata-sections", "-ffunction-sections",
              "-I", str(args.runtime_include), "-I", str(clone),
              "-isystem", str(include), "-isystem", str(args.libcxx_include),
              "-isystem", str(args.config_site_dir), "-DRADIANCE", "-DRADIANCE_DEVICE",
              "-DNDEBUG", "-DLLVM_VORTEX"]
    runtime_cpp = Path(__file__).resolve().parents[1] / "runtime/muon_mlir_runtime.cpp"
    objects = {
        "data": ["-x", "assembler-with-cpp", "-c", "data.S"],
        "source": ["-x", "c++", "-std=c++20", "-c", "kernel.cpp"],
        "mlir": ["-x", "ir", "-c", "kernel.ll"],
        "main": ["-x", "c", "-c", "main.c"],
        "runtime": ["-x", "c++", "-std=c++20", "-c", str(runtime_cpp)],
    }
    for name, flags in objects.items():
        run(common + flags + ["-o", f"{name}.o"], cwd=clone,
            log=clone / f"compile_{name}.log")
    records = {}
    for variant, linked in (("source", ["source.o", "data.o"]),
                            ("mlir", ["mlir.o", "main.o", "runtime.o", "data.o"])):
        elf = clone / f"{variant}.radiance.elf"
        run(common + ["-nostdlib", "-nodefaultlibs", "-nostartfiles", "-fuse-ld=lld",
             f"-Wl,-Bstatic,-T,{args.linker_script},-z,norelro", *linked,
             str(args.runtime_archive), str(args.tohost), "-o", str(elf)],
            cwd=clone, log=clone / f"link_{variant}.log")
        addr = symbol(args.llvm_bin / "llvm-nm", elf, "spatter_dense")
        check_log = out / f"{variant}.cyclotron.log"
        result = run([str(args.cyclotron_check), str(args.cyclotron_config), str(elf),
                      addr, str(case["length"]), str(case["count"]), str(case["wrap"]),
                      str(case["delta"]), str(clone / "generated/pattern.bin"),
                      str(clone / "generated/source.bin"), "1" if args.timing else "0"],
                     cwd=args.cyclotron_config.parent, log=check_log)
        cycles = re.search(r"simulation finished after (\d+) cycles", result)
        verified = re.search(r"SPATTER_FULL_GATHER_RESULT elements=\d+ digest=([0-9a-f]{16})", result)
        if not cycles or not verified:
            raise RuntimeError(f"{variant}: incomplete Cyclotron result; see {check_log}")
        if verified.group(1) != prepared["expected_digest"]:
            raise RuntimeError(f"{variant}: digest differs from source Spatter reference")
        records[variant] = {"elf_sha256": digest(elf), "cyclotron_steps": int(cycles.group(1)),
                            "output_address": addr, "output_digest": verified.group(1)}
        print(f"{variant}: {case['dst_length']} exact Gather outputs and guards passed; "
              f"{cycles.group(1)} {'timing' if args.timing else 'functional'} steps", flush=True)
    ratio = records["mlir"]["cyclotron_steps"] / records["source"]["cyclotron_steps"]
    manifest = {
        "schema": "muon_mlir_spatter_full_gather.v1", "suite_sha256": digest(args.suite),
        "case": args.case, "kind": "gather", "length": case["length"],
        "count": case["count"], "wrap": case["wrap"], "delta": case["delta"],
        "source_kernel_sha256": digest(args.source / "kernel.cpp"),
        "source_run_sha256": digest(args.source / "run.py"),
        "source_plan_sha256": digest(args.source / "plan.py"),
        "source_transfer_ops_sha256": digest(args.source / "spatter_ops.hpp"),
        "source_data_sha256": digest(clone / "generated/source.bin"),
        "pattern_sha256": digest(clone / "generated/pattern.bin"),
        "data_assembly_sha256": digest(clone / "data.S"),
        "mlir_sha256": digest(mlir), "llvm_ir_sha256": digest(ir),
        "compiler_sha256": digest(args.muon_clang),
        "runtime_archive_sha256": digest(args.runtime_archive),
        "cyclotron_config_sha256": digest(args.cyclotron_config),
        "cyclotron_checker_sha256": digest(args.cyclotron_check),
        "timing": args.timing, "runs": records,
        "ratio": ratio if args.timing else None,
        "within_10_percent": 0.90 <= ratio <= 1.10 if args.timing else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.timing and not manifest["within_10_percent"]:
        raise SystemExit("Muon MLIR exceeded the 10% Gather timing gate; see manifest.json")
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
