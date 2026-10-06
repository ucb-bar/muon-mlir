# Local source equivalence review

The source reference is the local `radiance-kernels-firesim` checkout on
`spatter-workloads`. The benchmark manifests pin the source and input bytes;
they do not assume that old FireSim logs are compiler results.

## STREAM

`kernels/stream/kernel.cpp` assigns each callback lane `block_id *
threads_per_block + tid` and advances by `MU_NUM_CLUSTERS *
threads_per_block`. The generated full-size MLIR uses the same start and
step for the selected single-cluster profile. Copy, Scale, Add, and Triad
load and store the same `stream_a/b/c` ELF symbols and perform the same FP32
operations. The output and input ELF sections are assembled once per case
and linked into both device variants. The generated entry adds a two-core
barrier before its fence to guarantee completed output readback.

The exact-input Cyclotron check at 1,048,576 elements compares every FP32
word and both guards. The cycle pairs in
`evidence/stream-full-timing-20261004.json` differ by at most 0.049%.
Both variants also fuse with the original `kernels/stream/host.cpp` and
`soc/fuse_rv32_into_rv64.sh`; see
`evidence/stream-soc-build-20261004.json`. These images still need an FPGA
guest run.

## Spatter Gather

For `gpu-stream.json` case 0, the original normalized case is length 256,
count 1024, wrap 1, and delta 256. The original `kernels/spatter/kernel.cpp`
assigns one lane to each dense output owner. It derives `r = owner / length`
and `j = owner - r * length`, then visits `i = r, r + wrap, ...` in order.
The generated MLIR has the same owner and iteration loops. The source helper
`spatter::affine_index(pattern[j], delta, i)` is lowered as `pattern[j] +
delta * i`, and the destination remains `owner`. Both implementations
perform two volatile 32-bit loads and two volatile 32-bit stores per 64-bit
transfer. The source's outer-loop overflow guard and the generated
less-than loop are equivalent over this bounded case. The selected profile
has one cluster, so both use `threads_per_block` as the lane stride.

The original `run.py` generates the pattern and payload blobs in a separate
source copy. Both device variants link the same data object. Cyclotron
compared all 256 final 64-bit output words and both guards against those
blobs. The handwritten and MLIR runs took 1,055,902 and 1,055,904 timing
steps, respectively; the byte hashes are in
`evidence/spatter-gpu-stream-gather-20261004.json`.

## Current boundary

Small target smoke cases cover the five Spatter families and verify every
final output. The original-size parametric lowering and comparison is
implemented for Gather. Scatter, GS, MultiGather, and MultiScatter still use
expanded tables in the small smoke path. Mixed MX/Muon IR has profile and
handoff verification in `radiance-mlir`; MX command execution is not yet
lowered from that IR.
