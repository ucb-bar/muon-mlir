"""Build one generated FP32 STREAM case into a Muon RV32 ELF."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from generate_stream import KINDS, N, generate


def run(*argv: str) -> None:
    subprocess.run(argv, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    generated_include = args.out / "include"
    generated_include.mkdir(exist_ok=True)
    assertion_handler = (args.libcxx_include.parent /
                         "vendor/llvm/default_assertion_handler.in")
    (generated_include / "__assertion_handler").write_bytes(
        assertion_handler.read_bytes())
    mlir = args.out / f"stream_{args.kind}.mlir"
    lowered = args.out / f"stream_{args.kind}.lowered.mlir"
    llvm_mlir = args.out / f"stream_{args.kind}.llvm.mlir"
    llvm_ir = args.out / f"stream_{args.kind}.ll"
    elf = args.out / f"stream_{args.kind}.radiance.elf"
    text, _ = generate(args.kind)
    mlir.write_text(text)
    run(str(args.muon_opt), "--lower-muon-runtime", str(mlir), "-o", str(lowered))
    run(str(args.llvm_bin / "mlir-opt"), str(lowered),
        "--convert-scf-to-cf", "--convert-arith-to-llvm",
        "--finalize-memref-to-llvm", "--convert-func-to-llvm",
        "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
        "-o", str(llvm_mlir))
    run(str(args.llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
        str(llvm_mlir), "-o", str(llvm_ir))
    # LLVM 23 spells an optional GEP no-wrap promise that the pinned Muon
    # LLVM 18 parser does not yet know. Dropping only that promise preserves
    # the address calculation while keeping the rest of the IR untouched.
    ir_text = llvm_ir.read_text()
    llvm_ir.write_text(ir_text.replace("getelementptr inbounds nuw ",
                                       "getelementptr inbounds "))
    selected = {"copy": "c", "scale": "b", "add": "c", "triad": "a"}[args.kind]
    formula = {
        "copy": "i + 1",
        "scale": "2 * (3 * i + 2)",
        "add": "(i + 1) + (2 * i + 1)",
        "triad": "(2 * i + 1) + 2 * (3 * i + 2)",
    }[args.kind]
    wrapper = args.out / "target_main.c"
    wrapper.write_text(
        "extern void entry(void);\n"
        f"extern float get_{selected}(int);\n"
        "volatile unsigned target_result[2] = {0, 0};\n"
        "int main(void) {\n"
        "  unsigned core;\n"
        "  __asm__ volatile (\"csrr %0, 0xCC2\" : \"=r\"(core));\n"
        "  if (core >= 2) return 1;\n"
        "  entry();\n"
        f"  for (int i = 0; i < {N}; ++i) {{\n"
        f"    if (get_{selected}(i) != (float)({formula})) {{\n"
        "      target_result[core] = 0xBAD00001u;\n"
        "      return 1;\n"
        "    }\n"
        "  }\n"
        "  target_result[core] = 0xC0DEFACEu;\n"
        "  return 0;\n"
        "}\n"
    )
    clang = str(args.muon_clang)
    common = [clang, "-target", "riscv32-unknown-elf",
              "-march=rv32im_zfinx_zhinx", "-mabi=ilp32",
              "-Xclang", "-target-feature", "-Xclang", "+vortex",
              "-O2", "-mcmodel=medany", "-ffreestanding", "-fno-builtin",
              "-fno-exceptions", "-fno-rtti", "-I", str(args.runtime_include),
              "-isystem", str(generated_include),
              "-isystem", str(args.libcxx_include), "-isystem", str(args.config_site_dir)]
    wrapper_cpp = Path(__file__).resolve().parents[1] / "runtime/muon_mlir_runtime.cpp"
    run(*common, "-x", "ir", "-c", str(llvm_ir), "-o", str(args.out / "kernel.o"))
    run(*common, "-x", "c", "-c", str(wrapper), "-o", str(args.out / "target_main.o"))
    run(*common, "-x", "c++", "-c", str(wrapper_cpp),
        "-o", str(args.out / "muon_mlir_runtime.o"))
    run(*common, "-nostdlib", "-nodefaultlibs", "-nostartfiles", "-fuse-ld=lld",
        f"-Wl,-T,{args.linker_script},-z,norelro",
        str(args.out / "kernel.o"), str(args.out / "target_main.o"),
        str(args.out / "muon_mlir_runtime.o"), str(args.runtime_archive),
        str(args.tohost), "-o", str(elf))
    print(elf)


if __name__ == "__main__":
    main()
