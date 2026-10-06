// A small executable STREAM copy.  The scheduler is supplied by either
// libmuonrt.a on Radiance or tests/host_runtime.c on the host.
module {
  memref.global "private" @src : memref<64xi64> = uninitialized
  memref.global "private" @dst : memref<64xi64> = uninitialized

  func.func @kernel(%arg: !llvm.ptr, %callback_tid: i32,
                    %callback_tpb: i32, %block_id: i32) {
    %tid = "muon.thread_id"() : () -> i32
    %tpb = "muon.threads_per_block"() : () -> i32
    %start = arith.index_cast %tid : i32 to index
    %step = arith.index_cast %tpb : i32 to index
    %limit = arith.constant 64 : index
    %src = memref.get_global @src : memref<64xi64>
    %dst = memref.get_global @dst : memref<64xi64>
    scf.for %i = %start to %limit step %step {
      %v = memref.load %src[%i] : memref<64xi64>
      memref.store %v, %dst[%i] : memref<64xi64>
    }
    return
  }

  func.func @entry() -> i64 {
    %zero = arith.constant 0 : index
    %limit = arith.constant 64 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : i64
    %sum0 = arith.constant 0 : i64
    %src = memref.get_global @src : memref<64xi64>
    %dst = memref.get_global @dst : memref<64xi64>
    scf.for %i = %zero to %limit step %one {
      %wide = arith.index_cast %i : index to i64
      %v = arith.muli %wide, %three : i64
      memref.store %v, %src[%i] : memref<64xi64>
      memref.store %sum0, %dst[%i] : memref<64xi64>
    }
    %null = llvm.mlir.zero : !llvm.ptr
    "muon.launch"(%null) {kernel = @kernel, warps_per_core = 1 : i32} : (!llvm.ptr) -> ()
    %sum = scf.for %i = %zero to %limit step %one iter_args(%acc = %sum0) -> (i64) {
      %v = memref.load %dst[%i] : memref<64xi64>
      %next = arith.addi %acc, %v : i64
      scf.yield %next : i64
    }
    return %sum : i64
  }
}
