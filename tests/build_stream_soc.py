"""Fuse full-size STREAM Muon ELFs with the original RV64 FireSim host check."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    with log.open("w") as output:
        process = subprocess.run(argv, cwd=cwd, env=env, stdout=output,
                                 stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"{argv[0]} exited {process.returncode}; see {log}\n"
                           f"{log.read_text()[-2500:]}")


def host_config(source: Path, kind: str, elements: int) -> str:
    spec = importlib.util.spec_from_file_location("radiance_stream_source", source / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = min(64, elements) if elements > 1024 else 0
    positions = ([i * (elements - 1) // (samples - 1) for i in range(samples)]
                 if samples else range(elements))
    expected = module.digest(module.float_word(module.result_value(kind, i))
                             for i in positions)
    return (f"#pragma once\n#define STREAM_KIND {module.KINDS[kind]}\n"
            f"#define STREAM_ELEMENTS {elements}u\n"
            f"#define STREAM_READBACK_SAMPLES {samples}u\n"
            f"#define STREAM_EXPECTED_READBACK_DIGEST 0x{expected:016x}ULL\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="manifest.json from run_stream_full.py")
    parser.add_argument("--source", type=Path, required=True,
                        help="radiance-kernels/kernels/stream")
    parser.add_argument("--soc-root", type=Path, required=True,
                        help="radiance-kernels/soc")
    parser.add_argument("--runtime-include", type=Path, required=True)
    parser.add_argument("--llvm-nm", type=Path, required=True)
    parser.add_argument("--riscv64-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("manifest", "source", "soc_root", "runtime_include",
                 "llvm_nm", "riscv64_bin", "out"):
        setattr(args, name, getattr(args, name).resolve())
    benchmark = json.loads(args.manifest.read_text())
    if benchmark.get("schema") != "muon_mlir_stream_full.v1":
        parser.error("--manifest must be a full STREAM benchmark manifest")
    if any(not case.get("runs") for case in benchmark["cases"]):
        parser.error("every STREAM case must have passed its target check")
    args.out.mkdir(parents=True, exist_ok=True)
    gcc = args.riscv64_bin / "riscv64-unknown-elf-gcc"
    gxx = args.riscv64_bin / "riscv64-unknown-elf-g++"
    ld = args.riscv64_bin / "riscv64-unknown-elf-ld"
    objcopy = args.riscv64_bin / "riscv64-unknown-elf-objcopy"
    records = []
    for case in benchmark["cases"]:
        kind, elements = case["kind"], case["elements"]
        case_root = args.manifest.parent / kind
        config = host_config(args.source, kind, elements)
        results = {}
        for variant in ("source", "mlir"):
            elf = case_root / f"{variant}.radiance.elf"
            if digest(elf) != case["runs"][variant]["elf_sha256"]:
                raise RuntimeError(f"{elf}: ELF differs from verified benchmark")
            work = args.out / kind / variant
            generated = work / "generated"
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "config.h").write_text(config)
            shutil.copy2(args.source / "host.cpp", work / "host.cpp")
            run(["python3", str(args.source / "emit_symbols.py"), str(args.llvm_nm),
                 str(elf), str(generated / "symbols.h")], cwd=work,
                log=work / "symbols.log")
            flags = ["-march=rv64imafd", "-mabi=lp64d", "-mcmodel=medany",
                     "-ffreestanding", "-fno-common", "-fno-builtin-printf", "-O3",
                     "-I", str(args.runtime_include)]
            run([str(gxx), *flags, "-c", "host.cpp", "-o", "host.o"], cwd=work,
                log=work / "host_compile.log")
            fused = work / "kernel.soc.elf"
            env = os.environ.copy()
            env.update({
                "CC": str(gcc), "LD": str(ld), "OBJCOPY": str(objcopy),
                "READELF": "readelf", "RV64_LINK": str(gcc),
                "RV64_START": str(args.soc_root / "start.S"), "RV64_MAIN": "",
                "RV64_OBJS": "host.o", "RV32_ELF": str(elf), "OUT": str(fused),
                "RV64_CFLAGS": "-march=rv64imafd -mabi=lp64d -ffreestanding -nostdlib -mcmodel=medany",
                "RV64_LDFLAGS": "-static -specs=htif_nano.specs",
            })
            run([str(args.soc_root / "fuse_rv32_into_rv64.sh")], cwd=work,
                log=work / "fuse.log", env=env)
            results[variant] = {
                "device_elf_sha256": digest(elf),
                "soc_elf_sha256": digest(fused),
                "host_source_sha256": digest(args.source / "host.cpp"),
                "host_object_sha256": digest(work / "host.o"),
                "symbols_sha256": digest(generated / "symbols.h"),
            }
            print(f"{kind} {variant}: fused RV64/RV32 image {fused}", flush=True)
        records.append({"kind": kind, "elements": elements, "images": results})
    output = args.out / "manifest.json"
    output.write_text(json.dumps({
        "schema": "muon_mlir_stream_soc_build.v1",
        "source_benchmark_sha256": digest(args.manifest),
        "host_compiler_sha256": digest(gxx),
        "fuser_sha256": digest(args.soc_root / "fuse_rv32_into_rv64.sh"),
        "cases": records,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
