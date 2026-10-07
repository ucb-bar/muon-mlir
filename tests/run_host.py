"""Build and execute the Muon MLIR STREAM and Spatter smoke cases on a host."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

from generate_spatter import address, generate
from generate_stream import KINDS as STREAM_KINDS, generate as generate_stream

HERE = Path(__file__).resolve().parent
CASES = [json.loads(path.read_text()) for path in sorted((HERE / "cases").glob("*.json"))]


def run(*argv: str) -> None:
    subprocess.run(argv, check=True)


def reference_check(cases: list[dict], plan_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("radiance_spatter_plan", plan_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {plan_path}")
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for case in cases:
        plan = module.plan_for(case)
        reads, writes = [], []
        for iteration in range(case["count"]):
            for j in range(case["length"]):
                assert address(case, "read", iteration, j) == plan.read.at(case, iteration, j)
                assert address(case, "write", iteration, j) == plan.write.at(case, iteration, j)
                reads.append(address(case, "read", iteration, j))
                writes.append(address(case, "write", iteration, j))
        source = [i * 3 + 7 for i in range(max(reads) + 1)]
        expected = [0] * (max(writes) + 1)
        for read, write in zip(reads, writes):
            expected[write] = source[read]
        oracle_case = dict(case, _plan=plan, src_length=len(source),
                           dst_length=len(expected))
        assert module.execute_reference(oracle_case, source) == expected


def compile_case(mlir: Path, harness: Path, stem: Path, muon_opt: Path,
                 llvm_bin: Path, distribute: bool = False,
                 blocks: int = 1) -> None:
    lowered = stem.with_suffix(".lowered.mlir")
    llvm_mlir = stem.with_suffix(".llvm.mlir")
    llvm_ir = stem.with_suffix(".ll")
    executable = stem.with_suffix(".exe")
    passes = ([f"--distribute-scf-parallel-to-muon=blocks={blocks}"]
              if distribute else [])
    run(str(muon_opt), *passes, "--lower-muon-runtime", str(mlir), "-o", str(lowered))
    if distribute and "scf.parallel" in lowered.read_text():
        raise RuntimeError("Muon distribution left a parallel loop in the callback")
    run(str(llvm_bin / "mlir-opt"), str(lowered),
        "--convert-scf-to-cf", "--convert-arith-to-llvm",
        "--finalize-memref-to-llvm", "--convert-func-to-llvm",
        "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
        "-o", str(llvm_mlir))
    run(str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
        str(llvm_mlir), "-o", str(llvm_ir))
    run(str(llvm_bin / "clang"), "-Wno-override-module", str(llvm_ir),
        str(harness), "-o", str(executable))
    run(str(executable))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--radiance-plan", type=Path)
    args = parser.parse_args()
    if args.radiance_plan:
        reference_check(CASES, args.radiance_plan)
    with tempfile.TemporaryDirectory(prefix="muon-mlir-host-") as directory:
        root = Path(directory)
        compile_case(HERE / "stream_copy.mlir", HERE / "host_runtime.c",
                     root / "stream_copy", args.muon_opt, args.llvm_bin)
        for kind in STREAM_KINDS:
            stem = root / f"stream_{kind}"
            mlir, harness = generate_stream(kind)
            stem.with_suffix(".mlir").write_text(mlir)
            stem.with_suffix(".c").write_text(harness)
            compile_case(stem.with_suffix(".mlir"), stem.with_suffix(".c"),
                         stem, args.muon_opt, args.llvm_bin)
            parallel = root / f"stream_parallel_{kind}"
            parallel_mlir, parallel_harness = generate_stream(
                kind, parallel=True, blocks=2)
            parallel.with_suffix(".mlir").write_text(parallel_mlir)
            parallel.with_suffix(".c").write_text(parallel_harness)
            compile_case(parallel.with_suffix(".mlir"), parallel.with_suffix(".c"),
                         parallel, args.muon_opt, args.llvm_bin,
                         distribute=True, blocks=2)
        compile_case(HERE / "parallel_nd.mlir", HERE / "parallel_nd_host.c",
                     root / "parallel_nd", args.muon_opt, args.llvm_bin,
                     distribute=True, blocks=2)
        for case in CASES:
            stem = root / case["kind"]
            mlir, harness = generate(case)
            stem.with_suffix(".mlir").write_text(mlir)
            stem.with_suffix(".c").write_text(harness)
            compile_case(stem.with_suffix(".mlir"), stem.with_suffix(".c"),
                         stem, args.muon_opt, args.llvm_bin)


if __name__ == "__main__":
    main()
