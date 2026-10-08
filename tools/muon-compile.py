"""Compile typed Muon MLIR with the matching Muon LLVM and runtime toolchain.

The input must provide its own entry point, or the caller must supply one with
--source/--extra-object for ELF output. This driver does not import handwritten
LLVM IR and does not claim that a simulator has executed the resulting image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_profile(path: Path, chipyard: Path | None) -> dict:
    if path.suffix == ".json":
        profile = json.loads(path.read_text())
    else:
        import yaml
        profile = yaml.safe_load(path.read_text())
    if not isinstance(profile, dict) or profile.get("schema") not in (
            "muon.target_profile.v1", "radiance.soc_profile.v1"):
        raise ValueError("expected a Muon or Radiance target profile")
    muon = profile.get("muon")
    if not isinstance(muon, dict) or muon.get("lanes_per_warp") != 16:
        raise ValueError("target profile lacks a supported 16-lane Muon topology")
    for key in ("clusters", "cores_per_cluster", "max_warps_per_core"):
        if not isinstance(muon.get(key), int) or muon[key] < 1:
            raise ValueError(f"target profile has invalid {key}")
    stride = muon.get("stack_word_stride", 16)
    if stride not in (1, 16):
        raise ValueError("supported stack-word strides are 1 and 16")
    if profile.get("source_files"):
        if chipyard is None:
            raise ValueError("--chipyard is required to bind the selected source profile")
        for relative, expected in profile["source_files"].items():
            actual = digest(chipyard / relative)
            if actual != expected:
                raise ValueError(f"profile source hash differs: {relative}")
    return profile


def run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{command[0]} exited {result.returncode}; see {log}")


def required(parser: argparse.ArgumentParser, name: str, value: Path | None) -> Path:
    if value is None:
        parser.error(f"{name} is required for the selected output")
    return value.resolve()


def probe_native_abi(clang: Path, stride: int) -> None:
    """Require the device compiler to accept the profile's stack ABI options."""
    options = (["-mllvm", f"-riscv-stack-word-stride={stride}"]
               if stride != 1 else [])
    result = subprocess.run(
        [str(clang), "-target", "riscv32-unknown-elf",
         "-march=rv32im_zfinx_zhinx", "-mabi=ilp32",
         "-Xclang", "-target-feature", "-Xclang", "+vortex",
         *options, "-ffreestanding", "-x", "c", "-c", "-", "-o", "/dev/null"],
        input="void muon_abi_probe(void) {}\n", capture_output=True, text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise ValueError(
            f"native Muon compiler does not support stack-word-stride={stride}: "
            f"{detail[0] if detail else 'compiler exited nonzero'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="typed Muon/standard-dialect MLIR module")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--chipyard", type=Path,
                        help="source checkout used to verify the profile's source hashes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--emit", choices=("llvm-ir", "object", "elf"), default="object")
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--mlir-bin", type=Path, required=True,
                        help="mlir-opt and mlir-translate from the native Muon LLVM build")
    parser.add_argument("--upstream-mlir-opt", type=Path,
                        help="newer upstream mlir-opt to normalize captured standard IR before native Muon LLVM")
    parser.add_argument("--muon-clang", type=Path)
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--runtime-include", type=Path)
    parser.add_argument("--libcxx-include", type=Path)
    parser.add_argument("--config-site-dir", type=Path)
    parser.add_argument("--linker-script", type=Path)
    parser.add_argument("--tohost", type=Path)
    parser.add_argument("--runtime-stack-word-stride", type=int, choices=(1, 16),
                        help="stride used to build the supplied runtime archive")
    parser.add_argument("--forward-inputs",
                        help="comma-separated external storage symbols for model2MLIR forward inputs")
    parser.add_argument("--forward-output",
                        help="external result storage symbol for model2MLIR forward")
    parser.add_argument("--forward-warps", type=int, default=4,
                        help="warps per core for an outlined model2MLIR forward")
    parser.add_argument("--forward-shared-scratch", action="store_true",
                        help="hoist intermediate allocations and synchronize callback stages")
    parser.add_argument("--source", type=Path, action="append", default=[],
                        help="additional C/C++ source, such as a handwritten test entry")
    parser.add_argument("--extra-object", type=Path, action="append", default=[])
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    work = (args.work_dir or output.parent / (output.name + ".build")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_path = args.profile.resolve()
    profile = load_profile(profile_path, args.chipyard.resolve() if args.chipyard else None)
    muon = profile["muon"]
    stride = muon.get("stack_word_stride", 16)
    muon_opt = args.muon_opt.resolve()
    mlir_bin = args.mlir_bin.resolve()
    clang = required(parser, "--muon-clang", args.muon_clang) if args.emit != "llvm-ir" else None
    if clang is not None:
        try:
            probe_native_abi(clang, stride)
        except ValueError as exc:
            parser.error(str(exc))
    if bool(args.forward_inputs) != bool(args.forward_output):
        parser.error("--forward-inputs and --forward-output must be supplied together")
    if args.forward_shared_scratch and (not args.forward_inputs or muon["clusters"] != 1):
        parser.error("shared scratch requires forward storage binding and one cluster")
    if args.forward_inputs and not 1 <= args.forward_warps <= muon["max_warps_per_core"]:
        parser.error("--forward-warps exceeds the selected profile")
    lowered = work / "muon.lowered.mlir"
    normalized = work / "muon.normalized.mlir"
    llvm_mlir = work / "muon.llvm.mlir"
    llvm_ir = work / "muon.ll"
    outline = ([f"--outline-forward-to-muon=inputs={args.forward_inputs} "
                f"output={args.forward_output} warps={args.forward_warps} "
                f"shared-scratch={'true' if args.forward_shared_scratch else 'false'}"]
               if args.forward_inputs else [])
    run([str(muon_opt), *outline,
         f"--distribute-scf-parallel-to-muon=blocks={muon['clusters']}",
         "--lower-muon-runtime", str(source), "-o", str(lowered)],
        work / "lower.log")
    launch_count = lowered.read_text().count("call @mu_schedule(")
    if args.emit != "llvm-ir" and launch_count == 0:
        parser.error("object/ELF output requires a Muon launch; distribute standard loops first")
    native_input = lowered
    if args.upstream_mlir_opt:
        run([str(args.upstream_mlir_opt.resolve()), str(lowered),
             "--expand-strided-metadata", "--lower-affine", "-o", str(normalized)],
            work / "normalize.log")
        native_input = normalized
    run([str(mlir_bin / "mlir-opt"), str(native_input),
         "--expand-strided-metadata", "--lower-affine", "--convert-scf-to-cf",
         "--convert-arith-to-llvm", "--finalize-memref-to-llvm",
         "--convert-func-to-llvm", "--convert-cf-to-llvm",
         "--reconcile-unrealized-casts", "-o", str(llvm_mlir)],
        work / "convert.log")
    run([str(mlir_bin / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir),
         "-o", str(llvm_ir)], work / "translate.log")
    if args.emit == "llvm-ir":
        shutil.copyfile(llvm_ir, output)
    else:
        assert clang is not None
        stack_flag = ["-mllvm", f"-riscv-stack-word-stride={stride}"] if stride != 1 else []
        common = [str(clang), "-target", "riscv32-unknown-elf",
                  "-march=rv32im_zfinx_zhinx", "-mabi=ilp32", "-Xclang",
                  "-target-feature", "-Xclang", "+vortex", "-O2", "-mcmodel=medany",
                  "-ffreestanding", "-fno-builtin", "-fno-exceptions", "-fno-rtti",
                  *stack_flag]
        kernel = work / "kernel.o"
        run([*common, "-x", "ir", "-c", str(llvm_ir), "-o", str(kernel)],
            work / "native.log")
        if args.emit == "object":
            shutil.copyfile(kernel, output)
        else:
            archive = required(parser, "--runtime-archive", args.runtime_archive)
            runtime_include = required(parser, "--runtime-include", args.runtime_include)
            libcxx = required(parser, "--libcxx-include", args.libcxx_include)
            config = required(parser, "--config-site-dir", args.config_site_dir)
            linker = required(parser, "--linker-script", args.linker_script)
            tohost = required(parser, "--tohost", args.tohost)
            if args.runtime_stack_word_stride != stride:
                parser.error("--runtime-stack-word-stride must match the selected profile")
            generated_include = work / "include"
            generated_include.mkdir(exist_ok=True)
            assertion_handler = libcxx.parent / "vendor/llvm/default_assertion_handler.in"
            (generated_include / "__assertion_handler").write_bytes(
                assertion_handler.read_bytes())
            headers = ["-I", str(runtime_include), "-isystem", str(generated_include),
                       "-isystem", str(libcxx), "-isystem", str(config)]
            runtime_source = Path(__file__).resolve().parents[1] / "runtime/muon_mlir_runtime.cpp"
            sources = [runtime_source, *(item.resolve() for item in args.source)]
            objects = [kernel]
            for index, item in enumerate(sources):
                object_path = work / f"source_{index}.o"
                language = "c" if item.suffix == ".c" else "c++"
                run([*common, *headers, "-x", language, "-c", str(item),
                     "-o", str(object_path)], work / f"source_{index}.log")
                objects.append(object_path)
            objects.extend(item.resolve() for item in args.extra_object)
            run([str(clang), "-target", "riscv32-unknown-elf",
                 "-march=rv32im_zfinx_zhinx", "-mabi=ilp32", "-Xclang",
                 "-target-feature", "-Xclang", "+vortex", "-mcmodel=medany",
                 "-nostdlib", "-nodefaultlibs", "-nostartfiles", "-fuse-ld=lld",
                 f"-Wl,-T,{linker},-z,norelro", *(str(item) for item in objects),
                 str(archive), str(tohost), "-o", str(output)], work / "link.log")
    receipt = {"schema": "muon_mlir_compile_receipt.v1",
               "status": "emitted_unexecuted", "emit": args.emit,
               "profile_name": profile["name"], "profile_sha256": digest(profile_path),
               "input_sha256": digest(source), "lowered_mlir_sha256": digest(lowered),
               "normalized_mlir_sha256": (digest(normalized) if args.upstream_mlir_opt
                                            else None),
               "llvm_ir_sha256": digest(llvm_ir), "output_sha256": digest(output),
               "muon_opt_sha256": digest(muon_opt),
               "mlir_translate_sha256": digest(mlir_bin / "mlir-translate"),
               "upstream_mlir_opt_sha256": (digest(args.upstream_mlir_opt.resolve())
                                             if args.upstream_mlir_opt else None),
               "stack_word_stride": stride,
               "muon_blocks": muon["clusters"],
               "forward_storage_binding": (
                   {"inputs": args.forward_inputs.split(","),
                    "output": args.forward_output, "warps_per_core": args.forward_warps,
                    "shared_scratch": args.forward_shared_scratch}
                   if args.forward_inputs else None),
               "muon_launch_count": launch_count,
               "scheduling_status": ("launch_lowered" if launch_count else
                                     "standard_ir_not_distributed")}
    if clang:
        receipt["muon_clang_sha256"] = digest(clang)
    if args.emit == "elf":
        receipt["runtime_archive_sha256"] = digest(archive)
        receipt["linker_script_sha256"] = digest(linker)
        receipt["tohost_sha256"] = digest(tohost)
    destination = output.with_name(output.name + ".receipt.json")
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(output), "receipt": str(destination),
                      "sha256": receipt["output_sha256"]}))


if __name__ == "__main__":
    main()
