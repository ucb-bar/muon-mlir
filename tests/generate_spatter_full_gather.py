"""Generate parametric Muon MLIR for the source Spatter Gather address map."""
from __future__ import annotations


def generate(case: dict) -> str:
    if case["kind"] != "gather":
        raise ValueError("this lowering supports Gather only")
    length, count, wrap, delta = (case[key] for key in ("length", "count", "wrap", "delta"))
    return f"""module {{
  llvm.mlir.global external @spatter_pattern() : i32
  llvm.mlir.global external @spatter_sparse() : i32
  llvm.mlir.global external @spatter_dense() : i32
  func.func @kernel(%arg: !llvm.ptr, %callback_tid: i32,
                    %callback_tpb: i32, %callback_bid: i32) {{
    %tid = "muon.thread_id"() : () -> i32
    %tpb = "muon.threads_per_block"() : () -> i32
    %bid = "muon.block_id"() : () -> i32
    %base = arith.muli %bid, %tpb : i32
    %global_tid = arith.addi %base, %tid : i32
    %length = arith.constant {length} : i32
    %owners = arith.constant {length * wrap} : i32
    %count = arith.constant {count} : i32
    %wrap = arith.constant {wrap} : i32
    %delta = arith.constant {delta} : i32
    %two = arith.constant 2 : i32
    %one = arith.constant 1 : i32
    %pattern = llvm.mlir.addressof @spatter_pattern : !llvm.ptr
    %sparse = llvm.mlir.addressof @spatter_sparse : !llvm.ptr
    %dense = llvm.mlir.addressof @spatter_dense : !llvm.ptr
    %outer = scf.while (%owner = %global_tid) : (i32) -> i32 {{
      %active = arith.cmpi ult, %owner, %owners : i32
      scf.condition(%active) %owner : i32
    }} do {{
    ^bb0(%owner: i32):
      %r = arith.divui %owner, %length : i32
      %rbase = arith.muli %r, %length : i32
      %j = arith.subi %owner, %rbase : i32
      %pp = llvm.getelementptr %pattern[%j] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %pattern_base = llvm.load %pp : !llvm.ptr -> i32
      %dst_word = arith.muli %owner, %two : i32
      %dst_high = arith.addi %dst_word, %one : i32
      %dp0 = llvm.getelementptr %dense[%dst_word] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %dp1 = llvm.getelementptr %dense[%dst_high] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %inner = scf.while (%i = %r) : (i32) -> i32 {{
        %active_inner = arith.cmpi ult, %i, %count : i32
        scf.condition(%active_inner) %i : i32
      }} do {{
      ^bb0(%i: i32):
        %offset = arith.muli %delta, %i : i32
        %src_index = arith.addi %pattern_base, %offset : i32
        %src_word = arith.muli %src_index, %two : i32
        %src_high = arith.addi %src_word, %one : i32
        %sp0 = llvm.getelementptr %sparse[%src_word] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %sp1 = llvm.getelementptr %sparse[%src_high] : (!llvm.ptr, i32) -> !llvm.ptr, i32
        %low = llvm.load volatile %sp0 : !llvm.ptr -> i32
        %high = llvm.load volatile %sp1 : !llvm.ptr -> i32
        llvm.store volatile %low, %dp0 : i32, !llvm.ptr
        llvm.store volatile %high, %dp1 : i32, !llvm.ptr
        %next_i = arith.addi %i, %wrap : i32
        scf.yield %next_i : i32
      }}
      %next_owner = arith.addi %owner, %tpb : i32
      scf.yield %next_owner : i32
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
