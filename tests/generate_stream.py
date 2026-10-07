"""Generate small FP32 STREAM kernels with the radiance-kernels operation maps."""
from __future__ import annotations

N = 97
KINDS = ("copy", "scale", "add", "triad")


def generate(kind: str, parallel: bool = False, blocks: int = 1) -> tuple[str, str]:
    if kind not in KINDS:
        raise ValueError(kind)
    if blocks < 1 or (blocks != 1 and not parallel):
        raise ValueError("multiple blocks require the parallel distribution pass")
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
    loop_setup = ("%zero = arith.constant 0 : index\n    "
                  "%one = arith.constant 1 : index\n    ") if parallel else ""
    loop_start = ("scf.parallel (%i) = (%zero) to (%limit) step (%one)"
                  if parallel else "scf.for %i = %start to %limit step %step")
    loop_end = "\n      scf.reduce" if parallel else ""
    counts_global = ("  memref.global @write_count : memref<97xi32> = dense<["
                     + ", ".join("0" for _ in range(N)) + "]>\n") if parallel else ""
    counter_ops = ("\n      %counts = memref.get_global @write_count : memref<97xi32>"
                   "\n      %old = memref.load %counts[%i] : memref<97xi32>"
                   "\n      %one_i32 = arith.constant 1 : i32"
                   "\n      %next = arith.addi %old, %one_i32 : i32"
                   "\n      memref.store %next, %counts[%i] : memref<97xi32>") if parallel else ""
    counter_getter = """  func.func @get_write_count(%i: i32) -> i32 {
    %index = arith.index_cast %i : i32 to index
    %counts = memref.get_global @write_count : memref<97xi32>
    %v = memref.load %counts[%index] : memref<97xi32>
    return %v : i32
  }
""" if parallel else ""
    mlir = f"""module {{
  memref.global @a : memref<97xf32> = dense<[{initial_a}]>
  memref.global @b : memref<97xf32> = dense<[{initial_b}]>
  memref.global @c : memref<97xf32> = dense<[{initial_c}]>
{counts_global}
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
    {loop_setup}{loop_start} {{
      {operation}{counter_ops}{loop_end}
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
{counter_getter}
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
    counter_decl = "extern int32_t get_write_count(int32_t index);" if parallel else ""
    counter_check = (f"""    if (get_write_count(i) != 1) {{
      fprintf(stderr, "{kind} output[%d] written %d times\\n", i, get_write_count(i));
      return 1;
    }}
""") if parallel else ""
    harness = f"""#include <stdint.h>
#include <stdio.h>
typedef void (*callback_t)(void *, uint32_t, uint32_t, uint32_t);
void mu_schedule(callback_t fn, void *arg, uint32_t occupancy) {{
  uint32_t threads = occupancy * 16;
  for (uint32_t bid = 0; bid < {blocks}; ++bid)
    for (uint32_t tid = 0; tid < threads; ++tid) fn(arg, tid, threads, bid);
}}
void muon_mlir_fence(void) {{}}
void muon_mlir_barrier(uint32_t id, uint32_t warps) {{ (void)id; (void)warps; }}
extern void entry(void);
extern float get_{selected}(int32_t index);
{counter_decl}
int main(void) {{
  static const float expected[97] = {{{values}}};
  entry();
  for (int i = 0; i < 97; ++i) {{
    float actual = get_{selected}(i);
    if (actual != expected[i]) {{
      fprintf(stderr, "{kind} output[%d] = %f, expected %f\\n", i, actual, expected[i]);
      return 1;
    }}
{counter_check}
  }}
  puts("STREAM {kind}: 97 exact FP32 outputs passed");
  return 0;
}}
"""
    return mlir, harness
