"""Generate the original-size STREAM operation over the source suite's ELF symbols."""
from __future__ import annotations

from generate_stream import KINDS


def generate(kind: str, elements: int) -> str:
    if kind not in KINDS or elements < 2:
        raise ValueError((kind, elements))
    operation = {
        "copy": """%ap = llvm.getelementptr %a[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %value = llvm.load %ap : !llvm.ptr -> f32
      %cp = llvm.getelementptr %c[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %value, %cp : f32, !llvm.ptr""",
        "scale": """%cp = llvm.getelementptr %c[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %value = llvm.load %cp : !llvm.ptr -> f32
      %scaled = arith.mulf %value, %two : f32
      %bp = llvm.getelementptr %b[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %scaled, %bp : f32, !llvm.ptr""",
        "add": """%ap = llvm.getelementptr %a[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %bp = llvm.getelementptr %b[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %av = llvm.load %ap : !llvm.ptr -> f32
      %bv = llvm.load %bp : !llvm.ptr -> f32
      %sum = arith.addf %av, %bv : f32
      %cp = llvm.getelementptr %c[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %sum, %cp : f32, !llvm.ptr""",
        "triad": """%bp = llvm.getelementptr %b[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %cp = llvm.getelementptr %c[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %bv = llvm.load %bp : !llvm.ptr -> f32
      %cv = llvm.load %cp : !llvm.ptr -> f32
      %scaled = arith.mulf %cv, %two : f32
      %sum = arith.addf %bv, %scaled : f32
      %ap = llvm.getelementptr %a[%i32] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %sum, %ap : f32, !llvm.ptr""",
    }[kind]
    return f"""module {{
  llvm.mlir.global external @stream_a() : f32
  llvm.mlir.global external @stream_b() : f32
  llvm.mlir.global external @stream_c() : f32
  func.func @kernel(%arg: !llvm.ptr, %callback_tid: i32,
                    %callback_tpb: i32, %callback_bid: i32) {{
    %tid = "muon.thread_id"() : () -> i32
    %tpb = "muon.threads_per_block"() : () -> i32
    %bid = "muon.block_id"() : () -> i32
    %base = arith.muli %bid, %tpb : i32
    %global_tid = arith.addi %base, %tid : i32
    %start = arith.index_cast %global_tid : i32 to index
    %step = arith.index_cast %tpb : i32 to index
    %limit = arith.constant {elements} : index
    %two = arith.constant 2.0 : f32
    %a = llvm.mlir.addressof @stream_a : !llvm.ptr
    %b = llvm.mlir.addressof @stream_b : !llvm.ptr
    %c = llvm.mlir.addressof @stream_c : !llvm.ptr
    scf.for %i = %start to %limit step %step {{
      %i32 = arith.index_cast %i : index to i32
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
}}
"""
