"""Generate small MLIR transfer kernels from normalized Spatter case data.

The source and destination address equations match radiance-kernels/spatter/plan.py.
Each destination is assigned to one lane; writes to a repeated destination
remain in source order.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def address(case: dict, side: str, iteration: int, j: int) -> int:
    kind = case["kind"]
    length = case["length"]
    wrap = case["wrap"]
    dense = j + length * (iteration % wrap)
    if kind == "gather":
        return case["pattern"][j] + case["delta"] * iteration if side == "read" else dense
    if kind == "scatter":
        return dense if side == "read" else case["pattern"][j] + case["delta"] * iteration
    if kind == "gs":
        if side == "read":
            final_wrap = case.get("gather_final_wrap", 0)
            if final_wrap:
                residue = iteration % final_wrap
                iteration = residue + ((case["count"] - 1 - residue) // final_wrap) * final_wrap
            return case["gather"][j] + case["delta_gather"] * iteration
        return case["scatter"][j] + case["delta_scatter"] * iteration
    if kind == "multigather":
        return (case["pattern"][case["gather"][j]] + case["delta"] * iteration
                if side == "read" else dense)
    if kind == "multiscatter":
        return (dense if side == "read" else
                case["pattern"][case["scatter"][j]] + case["delta"] * iteration)
    raise ValueError(f"unknown transfer family: {kind}")


def global_i32(name: str, values: list[int]) -> str:
    contents = ", ".join(map(str, values))
    return f'  memref.global "private" constant @{name} : memref<{len(values)}xi32> = dense<[{contents}]>\n'


def generate(case: dict) -> tuple[str, str]:
    if any(not isinstance(case.get(key), int) or case[key] <= 0
           for key in ("count", "length", "wrap")):
        raise ValueError("count, length, and wrap must be positive integers")
    reads, writes = [], []
    for iteration in range(case["count"]):
        for j in range(case["length"]):
            reads.append(address(case, "read", iteration, j))
            writes.append(address(case, "write", iteration, j))
    if min(reads + writes) < 0:
        raise ValueError("negative address")
    if max(reads + writes) >= 2**31 or len(reads) >= 2**31:
        raise ValueError("expanded addresses exceed the i32 index table")
    src_len, dst_len = max(reads) + 1, max(writes) + 1
    source = [i * 3 + 7 for i in range(src_len)]
    expected = [0] * dst_len
    groups: dict[int, list[int]] = defaultdict(list)
    for ordinal, (read, write) in enumerate(zip(reads, writes)):
        expected[write] = source[read]
        groups[write].append(ordinal)
    tasks = [ordinal for destination in sorted(groups) for ordinal in groups[destination]]
    offsets = [0]
    for destination in sorted(groups):
        offsets.append(offsets[-1] + len(groups[destination]))
    n_groups = len(groups)
    source_values = ", ".join(map(str, source))
    destination_zeros = ", ".join("0" for _ in range(dst_len))
    text = f"""module {{
  memref.global @src : memref<{src_len}xi64> = dense<[{source_values}]>
  memref.global @dst : memref<{dst_len}xi64> = dense<[{destination_zeros}]>
{global_i32("read_map", reads)}{global_i32("write_map", writes)}{global_i32("tasks", tasks)}{global_i32("offsets", offsets)}
  func.func @kernel(%arg: !llvm.ptr, %callback_tid: i32,
                    %callback_tpb: i32, %block_id: i32) {{
    %tid = "muon.thread_id"() : () -> i32
    %tpb = "muon.threads_per_block"() : () -> i32
    %start = arith.index_cast %tid : i32 to index
    %step = arith.index_cast %tpb : i32 to index
    %limit = arith.constant {n_groups} : index
    %one = arith.constant 1 : index
    %src = memref.get_global @src : memref<{src_len}xi64>
    %dst = memref.get_global @dst : memref<{dst_len}xi64>
    %read_map = memref.get_global @read_map : memref<{len(reads)}xi32>
    %write_map = memref.get_global @write_map : memref<{len(writes)}xi32>
    %tasks = memref.get_global @tasks : memref<{len(tasks)}xi32>
    %offsets = memref.get_global @offsets : memref<{len(offsets)}xi32>
    scf.for %g = %start to %limit step %step {{
      %g_next = arith.addi %g, %one : index
      %begin_i32 = memref.load %offsets[%g] : memref<{len(offsets)}xi32>
      %end_i32 = memref.load %offsets[%g_next] : memref<{len(offsets)}xi32>
      %begin = arith.index_cast %begin_i32 : i32 to index
      %end = arith.index_cast %end_i32 : i32 to index
      scf.for %t = %begin to %end step %one {{
        %ordinal_i32 = memref.load %tasks[%t] : memref<{len(tasks)}xi32>
        %ordinal = arith.index_cast %ordinal_i32 : i32 to index
        %read_i32 = memref.load %read_map[%ordinal] : memref<{len(reads)}xi32>
        %write_i32 = memref.load %write_map[%ordinal] : memref<{len(writes)}xi32>
        %read = arith.index_cast %read_i32 : i32 to index
        %write = arith.index_cast %write_i32 : i32 to index
        %value = memref.load %src[%read] : memref<{src_len}xi64>
        memref.store %value, %dst[%write] : memref<{dst_len}xi64>
      }}
    }}
    return
  }}
  func.func @entry() {{
    %null = llvm.mlir.zero : !llvm.ptr
    "muon.launch"(%null) {{kernel = @kernel, warps_per_core = 1 : i32}} : (!llvm.ptr) -> ()
    "muon.barrier"() {{barrier_id = 0 : i32, num_warps = 2 : i32}} : () -> ()
    "muon.fence"() : () -> ()
    return
  }}
  func.func @get_output(%i: i32) -> i64 {{
    %index = arith.index_cast %i : i32 to index
    %dst = memref.get_global @dst : memref<{dst_len}xi64>
    %value = memref.load %dst[%index] : memref<{dst_len}xi64>
    return %value : i64
  }}
}}
"""
    expected_c = ", ".join(map(str, expected))
    harness = f"""#include <stdint.h>
#include <stdio.h>
typedef void (*callback_t)(void *, uint32_t, uint32_t, uint32_t);
void mu_schedule(callback_t fn, void *arg, uint32_t occupancy) {{
  uint32_t threads = occupancy * 16;
  for (uint32_t tid = 0; tid < threads; ++tid) fn(arg, tid, threads, 0);
}}
void muon_mlir_fence(void) {{}}
void muon_mlir_barrier(uint32_t id, uint32_t warps) {{ (void)id; (void)warps; }}
extern void entry(void);
extern int64_t get_output(int32_t index);
int main(void) {{
  static const int64_t expected[{dst_len}] = {{{expected_c}}};
  entry();
  for (int i = 0; i < {dst_len}; ++i) {{
    int64_t actual = get_output(i);
    if (actual != expected[i]) {{
      fprintf(stderr, "output[%d] = %lld, expected %lld\\n", i,
              (long long)actual, (long long)expected[i]);
      return 1;
    }}
  }}
  puts("{case['kind']}: {dst_len} exact outputs passed");
  return 0;
}}
"""
    return text, harness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    mlir, harness = generate(case)
    args.output_prefix.with_suffix(".mlir").write_text(mlir)
    args.output_prefix.with_suffix(".c").write_text(harness)


if __name__ == "__main__":
    main()
