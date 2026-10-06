"""Generate small FP32 STREAM kernels with the radiance-kernels operation maps."""
from __future__ import annotations

N = 97
KINDS = ("copy", "scale", "add", "triad")


def generate(kind: str) -> tuple[str, str]:
    if kind not in KINDS:
        raise ValueError(kind)
    operation = {
        "copy": """%a = memref.load %aa[%i] : memref<97xf32>
      memref.store %a, %cc[%i] : memref<97xf32>""",
        "scale": """%c = memref.load %cc[%i] : memref<97xf32>
      %v = arith.mulf %c, %two : f32
      memref.store %v, %bb[%i] : memref<97xf32>""",
        "add": """%a = memref.load %aa[%i] : memref<97xf32>
      %b = memref.load %bb[%i] : memref<97xf32>
      %v = arith.addf %a, %b : f32
      memref.store %v, %cc[%i] : memref<97xf32>""",
        "triad": """%b = memref.load %bb[%i] : memref<97xf32>
      %c = memref.load %cc[%i] : memref<97xf32>
      %scaled = arith.mulf %c, %two : f32
      %v = arith.addf %b, %scaled : f32
      memref.store %v, %aa[%i] : memref<97xf32>""",
    }[kind]
    initial_a = ", ".join(f"{i + 1}.0" for i in range(N))
    initial_b = ", ".join(f"{2 * i + 1}.0" for i in range(N))
    initial_c = ", ".join(f"{3 * i + 2}.0" for i in range(N))
    mlir = f"""module {{
  memref.global @a : memref<97xf32> = dense<[{initial_a}]>
  memref.global @b : memref<97xf32> = dense<[{initial_b}]>
  memref.global @c : memref<97xf32> = dense<[{initial_c}]>
  func.func @kernel(%arg: !llvm.ptr, %callback_tid: i32,
                    %callback_tpb: i32, %callback_bid: i32) {{
    %tid = "muon.thread_id"() : () -> i32
    %tpb = "muon.threads_per_block"() : () -> i32
    %bid = "muon.block_id"() : () -> i32
    %base = arith.muli %bid, %tpb : i32
    %global_tid = arith.addi %base, %tid : i32
    %start = arith.index_cast %global_tid : i32 to index
    %step = arith.index_cast %tpb : i32 to index
    %limit = arith.constant 97 : index
    %two = arith.constant 2.0 : f32
    %aa = memref.get_global @a : memref<97xf32>
    %bb = memref.get_global @b : memref<97xf32>
    %cc = memref.get_global @c : memref<97xf32>
    scf.for %i = %start to %limit step %step {{
      {operation}
    }}
    return
  }}
  func.func @entry() {{
    %null = llvm.mlir.zero : !llvm.ptr
    "muon.launch"(%null) {{kernel = @kernel, warps_per_core = 4 : i32}} : (!llvm.ptr) -> ()
    "muon.barrier"() {{barrier_id = 0 : i32, num_warps = 2 : i32}} : () -> ()
    "muon.fence"() : () -> ()
    return
  }}
  func.func @get_a(%i: i32) -> f32 {{
    %index = arith.index_cast %i : i32 to index
    %aa = memref.get_global @a : memref<97xf32>
    %v = memref.load %aa[%index] : memref<97xf32>
    return %v : f32
  }}
  func.func @get_b(%i: i32) -> f32 {{
    %index = arith.index_cast %i : i32 to index
    %bb = memref.get_global @b : memref<97xf32>
    %v = memref.load %bb[%index] : memref<97xf32>
    return %v : f32
  }}
  func.func @get_c(%i: i32) -> f32 {{
    %index = arith.index_cast %i : i32 to index
    %cc = memref.get_global @c : memref<97xf32>
    %v = memref.load %cc[%index] : memref<97xf32>
    return %v : f32
  }}
}}
"""
    expected = {
        "copy": ("c", lambda i: i + 1),
        "scale": ("b", lambda i: 2 * (3 * i + 2)),
        "add": ("c", lambda i: (i + 1) + (2 * i + 1)),
        "triad": ("a", lambda i: (2 * i + 1) + 2 * (3 * i + 2)),
    }
    selected, formula = expected[kind]
    values = ", ".join(f"{formula(i)}.0f" for i in range(N))
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
extern float get_{selected}(int32_t index);
int main(void) {{
  static const float expected[97] = {{{values}}};
  entry();
  for (int i = 0; i < 97; ++i) {{
    float actual = get_{selected}(i);
    if (actual != expected[i]) {{
      fprintf(stderr, "{kind} output[%d] = %f, expected %f\\n", i, actual, expected[i]);
      return 1;
    }}
  }}
  puts("STREAM {kind}: 97 exact FP32 outputs passed");
  return 0;
}}
"""
    return mlir, harness
