# Muon MLIR

Out-of-tree MLIR dialect for Muon SIMT execution. The dialect owns callback
launch, thread and block IDs, barriers, and memory fences. Ordinary indexed
loads, stores, loops, arithmetic, and reductions use upstream MLIR dialects.
The runtime lowering pass emits the existing `mu_schedule` ABI and wrapper
calls for Muon intrinsics; it does not introduce a second hardware scheduler.

The callback type is `(!llvm.ptr, i32 tid, i32 threads_per_block, i32
block_id) -> ()`. `muon.launch` carries the argument pointer and occupancy.
The verifier bounds occupancy to eight warps per core. Build every target
object and `libmuonrt.a` with a compatible Muon LLVM revision and the same
stack stride.

## Build and host smoke

```sh
cmake -S . -B build -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir
cmake --build build -j
python3 tests/run_host.py --muon-opt build/tools/muon-opt \
  --llvm-bin /path/to/llvm-install/bin \
  --radiance-plan /path/to/radiance-kernels/kernels/spatter/plan.py
```

`tests/stream_copy.mlir` is compiled through MLIR to LLVM IR and linked as a
host executable. The Spatter test generates all five transfer families as
MLIR and compares every output element. Repeated destinations are assigned to
one lane in original transfer order. The optional `--radiance-plan` check
compares every generated address to the source equations in
`radiance-kernels`.

The target smoke manifest `evidence/target-smoke-20261004.json` records
Cyclotron completion and exact final-output checks on both Muon cores for all
four STREAM operations and all five Spatter families.

These are correctness smoke cases at small sizes. The Spatter expanded index
tables are not a throughput lowering for the original large traces. A target ELF
requires native Muon LLVM, a compatible runtime archive, a SoC host image,
and simulator or FPGA readback. A generic RISC-V compiler can produce an ELF
but is incompatible with the Vortex runtime instruction encoding.

## Original-size STREAM comparison

`tests/run_stream_full.py` builds the handwritten `radiance-kernels` STREAM
source and generated Muon MLIR against the same source-generated input blobs,
same Muon compiler and runtime, and same Cyclotron configuration. It reads the
source `run.py` but writes only under `--out`. The checker compares **every**
FP32 output word and both 64-byte guards against the original input formulas.
With `--timing`, the manifest records both timing-model cycle counts and the
MLIR/source ratio for the same one-million-element case. This is a simulator
comparison; it does not establish FPGA timing or guest-host integration.

Build the checker with `tests/build_stream_full_check.py --cyclotron
/path/to/cyclotron --out build/stream_full_checker --target-dir
/path/to/cyclotron/target`. Then run `tests/run_stream_full.py --kind all
--elements 1048576 --timing --out build/stream_full_timing --source
/path/to/radiance-kernels/kernels/stream` with the explicit `--muon-opt`,
`--llvm-bin`, `--muon-clang`, `--runtime-archive`, `--runtime-include`,
`--libcxx-include`, `--config-site-dir`, `--linker-script`, `--tohost`,
`--cyclotron-check`, and `--cyclotron-config` paths. The generated LLVM IR
comes from upstream MLIR; the final ELF must use the native Muon compiler.

The 2026-10-04 pinned run is in `evidence/stream-full-timing-20261004.json`.
All 1,048,576 FP32 outputs and both guards passed for each variant:

| Operation | Handwritten cycles | Muon MLIR cycles | Difference |
| --- | ---: | ---: | ---: |
| Copy | 2,104,119 | 2,104,161 | +0.002% |
| Scale | 2,104,117 | 2,104,152 | +0.002% |
| Add | 3,270,851 | 3,269,252 | -0.049% |
| Triad | 3,270,855 | 3,270,528 | -0.010% |

The generated entry uses a two-core post-launch barrier followed by a fence;
the handwritten entry uses its original post-launch fence.

These are Cyclotron timing-model steps for freshly built pairs, not the
FireSim FPGA cycle counts in `radiance-kernels` logs. The evidence manifest
pins input, compiler, runtime, checker, ELF, and IR hashes.

`tests/build_stream_soc.py` takes that benchmark manifest, the original
`radiance-kernels/kernels/stream` host source, its `soc` fuse script, and an
RV64 cross compiler. It emits both handwritten and Muon MLIR fused RV64 host
plus RV32 device images without writing to the kernel checkout. The original
host checker uses the original 64-position readback digest and guard canaries.
All eight images built successfully; their hashes are in
`evidence/stream-soc-build-20261004.json`. This verifies packaging and
linking. FPGA guest execution is still required to establish host readback
and device completion on the selected bitstream.

## Original-size Spatter Gather

`tests/generate_spatter_full_gather.py` lowers the Gather address map as
runtime `i32` loops and volatile 64-bit transfers over the original ELF data
symbols. It follows the handwritten owner schedule: each dense output slot
has one lane, and that lane visits the matching iterations in order. The
source equation is `pattern[j] + delta * iteration`; the destination is
`j + length * (iteration % wrap)`. The source and generated objects each
contain two 32-bit loads and two 32-bit stores per transfer.

`tests/run_spatter_full_gather.py` makes a private copy of the local
`radiance-kernels` Spatter source, invokes its `run.py` for the chosen JSON
case, and compiles the handwritten and MLIR paths against the resulting
identical input and pattern blobs. Its checker compares every output word and
both guards against those blobs. For `gpu-stream.json` case 0 (256 by 1024),
the pinned manifest is `evidence/spatter-gpu-stream-gather-20261004.json`:
1,055,902 handwritten and 1,055,904 Muon MLIR Cyclotron timing steps, with
all 256 final output words equal. The other four Spatter families currently
have complete-output target smoke tests at small sizes; they have not yet
received this parametric original-size lowering.

The operation-by-operation comparison with local kernel source is in
`docs/source_equivalence.md`.

## Coverage inventory

`tools/inventory_kernels.py` reads the local kernel source and evaluates each
family's default Makefile target list without building it. The pinned
`evidence/kernel-inventory-20261006.json` records source and Makefile hashes
for revision `a27f6abd` of the clean radiance-kernels checkout. It finds 64
families, 195 source units, and 136 default Radiance ELF targets, including
20 named variants. Environment-driven sweeps can add further variants; the
inventory records the default configuration only. These counts define work
to verify, not current compiler coverage.

## Standalone profile and typed compiler entry

`profiles/radiance-muon-one-core.json` binds the one-core
`RadianceMuonConfig` to the selected Chipyard configuration source hash. The
experimental `tools/muon-compile.py` driver verifies that hash, lowers Muon
operations through `muon-opt`, translates with the matching native Muon MLIR
tools, and can emit LLVM IR, an RV32 object, or an ELF with explicit runtime
and linker inputs. Its receipt records the profile, tools, input, and output
hashes. `--emit llvm-ir` is currently verified on `tests/stream_copy.mlir`.
The available native Muon Clang predates the stack-word-stride option required
by the source's 16-stride profile, so an object or ELF from this configuration
is not yet qualified. The driver refuses a mismatched source hash or runtime
stride instead of silently compiling for another topology.

For the verified IR stage, run:

```sh
python3 tools/muon-compile.py tests/stream_copy.mlir \
  --profile profiles/radiance-muon-one-core.json \
  --chipyard /path/to/chipyard --emit llvm-ir \
  --muon-opt build/tools/muon-opt --mlir-bin /path/to/muon-llvm/bin \
  --output build/stream_copy.one_core.ll
```

The latest model2MLIR frontend capture is in `radiance-mlir`:
`tests/capture_model2mlir_stream.py` checks all four one-million-element
PyTorch STREAM outputs against the handwritten source equations, and
`tests/capture_model2mlir_gemm.py` checks all 4,096 BF16 SIMT GEMM output
words against the source generator. The resulting typed upstream MLIR parses
here. It still needs Muon launch/thread distribution and native target
lowering. The separate Radiance host check executes the generated STREAM
parallel loops and SIMT GEMM matmul against the complete source goldens;
that is not Muon device execution.

## Radiance composition

Radiance SoC facts and MX/Muon synchronization live in the separate
`radiance-mlir` package. MX operand encoding, contraction, and readout stay in
`mx-gemmini-mlir`. The three packages can be built together without adding
target-specific code to Merlin.
