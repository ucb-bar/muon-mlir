"""Build all nine HPC smoke ELFs and check complete outputs in Cyclotron."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from generate_stream import KINDS
from run_host import CASES, HERE, reference_check


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(argv: list[str], log: Path, *, cwd: Path | None = None) -> str:
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, cwd=cwd, env=env)
    log.write_text(proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(argv[:2])} failed; see {log}")
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--muon-clang", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-include", type=Path, required=True)
    parser.add_argument("--libcxx-include", type=Path, required=True)
    parser.add_argument("--config-site-dir", type=Path, required=True)
    parser.add_argument("--linker-script", type=Path, required=True)
    parser.add_argument("--tohost", type=Path, required=True)
    parser.add_argument("--radiance-plan", type=Path, required=True)
    parser.add_argument("--cyclotron-check", type=Path, required=True)
    parser.add_argument("--cyclotron-config", type=Path, required=True)
    args = parser.parse_args()
    reference_check(CASES, args.radiance_plan)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    common = ["--muon-opt", str(args.muon_opt), "--llvm-bin", str(args.llvm_bin),
              "--muon-clang", str(args.muon_clang),
              "--runtime-archive", str(args.runtime_archive),
              "--runtime-include", str(args.runtime_include),
              "--libcxx-include", str(args.libcxx_include),
              "--config-site-dir", str(args.config_site_dir),
              "--linker-script", str(args.linker_script), "--tohost", str(args.tohost)]
    records = []
    workloads = [("stream", kind, None) for kind in KINDS]
    workloads += [("spatter", case["kind"], HERE / "cases" / f"{case['kind']}.json")
                  for case in CASES]
    for family, kind, case_path in workloads:
        name = f"{family}_{kind}"
        directory = out / name
        directory.mkdir(exist_ok=True)
        if family == "stream":
            builder = HERE / "build_target_stream.py"
            argv = [sys.executable, str(builder), kind]
            elf = directory / f"stream_{kind}.radiance.elf"
            ir = directory / f"stream_{kind}.mlir"
        else:
            builder = HERE / "build_target_spatter.py"
            argv = [sys.executable, str(builder), str(case_path)]
            elf = directory / "spatter.radiance.elf"
            ir = directory / "spatter.mlir"
        execute(argv + ["--out", str(directory)] + common, directory / "build.log")
        nm = subprocess.run([str(args.llvm_bin / "llvm-nm"), "-n", str(elf)],
                            capture_output=True, text=True, check=True).stdout
        symbol = re.search(r"^([0-9a-fA-F]+) [A-Za-z] target_result$", nm, re.M)
        if not symbol:
            raise RuntimeError(f"{name}: no target_result symbol")
        result = execute([str(args.cyclotron_check), str(args.cyclotron_config),
                          str(elf), f"0x{symbol.group(1)}"],
                         directory / "cyclotron.log", cwd=args.cyclotron_config.parent)
        cycles = re.search(r"simulation finished after (\d+) cycles", result)
        status = re.search(
            r"MUON_MLIR_RESULT addr=0x[0-9a-f]+ core0=0xc0deface core1=0xc0deface",
            result)
        if not cycles or not status:
            raise RuntimeError(f"{name}: no verified Cyclotron completion")
        records.append({
            "family": family, "kind": kind, "status": "passed",
            "scope": "small_complete_output_smoke",
            "input_sha256": sha256(case_path) if case_path else sha256(ir),
            "mlir_sha256": sha256(ir), "elf_sha256": sha256(elf),
            "cyclotron_steps": int(cycles.group(1)),
            "result_address": f"0x{symbol.group(1)}",
        })
        print(f"{name}: exact output passed, {cycles.group(1)} Cyclotron steps")
    manifest = {
        "schema": "muon_mlir_target_smoke.v1",
        "radiance_plan_sha256": sha256(args.radiance_plan),
        "cyclotron_config_sha256": sha256(args.cyclotron_config),
        "cyclotron_check_sha256": sha256(args.cyclotron_check),
        "muon_clang_sha256": sha256(args.muon_clang),
        "runtime_archive_sha256": sha256(args.runtime_archive),
        "muon_opt_sha256": sha256(args.muon_opt),
        "workloads": records,
    }
    destination = out / "manifest.json"
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
