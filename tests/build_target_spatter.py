"""Build a normalized small Spatter case into a Muon RV32 ELF."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from generate_spatter import address, generate


def run(*argv: str) -> None:
    subprocess.run(argv, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
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
    case = json.loads(args.case.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    mlir = args.out / "spatter.mlir"
    lowered = args.out / "spatter.lowered.mlir"
    llvm_mlir = args.out / "spatter.llvm.mlir"
    llvm_ir = args.out / "spatter.ll"
    elf = args.out / "spatter.radiance.elf"
    text, _ = generate(case)
    mlir.write_text(text)
    run(str(args.muon_opt), "--lower-muon-runtime", str(mlir), "-o", str(lowered))
    run(str(args.llvm_bin / "mlir-opt"), str(lowered),
        "--convert-scf-to-cf", "--convert-arith-to-llvm",
        "--finalize-memref-to-llvm", "--convert-func-to-llvm",
        "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
        "-o", str(llvm_mlir))
    run(str(args.llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
        str(llvm_mlir), "-o", str(llvm_ir))
    llvm_ir.write_text(llvm_ir.read_text().replace(
        "getelementptr inbounds nuw ", "getelementptr inbounds "))
    reads = [address(case, "read", i, j)
             for i in range(case["count"]) for j in range(case["length"])]
    writes = [address(case, "write", i, j)
              for i in range(case["count"]) for j in range(case["length"])]
    source = [i * 3 + 7 for i in range(max(reads) + 1)]
    expected = [0] * (max(writes) + 1)
    for read, write in zip(reads, writes):
        expected[write] = source[read]
    wrapper = args.out / "target_main.c"
    values = ", ".join(f"{v}ULL" for v in expected)
    wrapper.write_text(
        "extern void entry(void);\n"
        "extern unsigned long long get_output(int);\n"
        "volatile unsigned target_result[2] = {0, 0};\n"
        f"static const unsigned long long expected[{len(expected)}] = {{{values}}};\n"
        "int main(void) {\n"
        "  unsigned core;\n"
        "  __asm__ volatile (\"csrr %0, 0xCC2\" : \"=r\"(core));\n"
        "  if (core >= 2) return 1;\n"
        "  entry();\n"
        f"  for (int i = 0; i < {len(expected)}; ++i) {{\n"
        "    if (get_output(i) != expected[i]) {\n"
        "      target_result[core] = 0xBAD00001u;\n"
        "      return 1;\n"
        "    }\n"
        "  }\n"
        "  target_result[core] = 0xC0DEFACEu;\n"
        "  return 0;\n"
        "}\n"
    )
    generated_include = args.out / "include"
    generated_include.mkdir(exist_ok=True)
    assertion_handler = (args.libcxx_include.parent /
                         "vendor/llvm/default_assertion_handler.in")
    (generated_include / "__assertion_handler").write_bytes(
        assertion_handler.read_bytes())
    common = [str(args.muon_clang), "-target", "riscv32-unknown-elf",
              "-march=rv32im_zfinx_zhinx", "-mabi=ilp32",
              "-Xclang", "-target-feature", "-Xclang", "+vortex",
              "-O2", "-mcmodel=medany", "-ffreestanding", "-fno-builtin",
              "-fno-exceptions", "-fno-rtti", "-I", str(args.runtime_include),
              "-isystem", str(generated_include), "-isystem", str(args.libcxx_include),
              "-isystem", str(args.config_site_dir)]
    runtime = Path(__file__).resolve().parents[1] / "runtime/muon_mlir_runtime.cpp"
    run(*common, "-x", "ir", "-c", str(llvm_ir), "-o", str(args.out / "kernel.o"))
    run(*common, "-x", "c", "-c", str(wrapper), "-o", str(args.out / "target_main.o"))
    run(*common, "-x", "c++", "-c", str(runtime),
        "-o", str(args.out / "muon_mlir_runtime.o"))
    run(*common, "-nostdlib", "-nodefaultlibs", "-nostartfiles", "-fuse-ld=lld",
        f"-Wl,-T,{args.linker_script},-z,norelro",
        str(args.out / "kernel.o"), str(args.out / "target_main.o"),
        str(args.out / "muon_mlir_runtime.o"), str(args.runtime_archive),
        str(args.tohost), "-o", str(elf))
    print(elf)


if __name__ == "__main__":
    main()
