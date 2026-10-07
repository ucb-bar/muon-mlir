// Pass regression: distribute the outer dimension, retain inner dimensions.
module {
  memref.global @counts : memref<3x4x2xi32> = dense<0>
  func.func @kernel(%arg: !llvm.ptr, %tid: i32, %threads: i32, %block: i32) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    %four = arith.constant 4 : index
    %two = arith.constant 2 : index
    %increment = arith.constant 1 : i32
    %counts = memref.get_global @counts : memref<3x4x2xi32>
    scf.parallel (%i, %j, %k) = (%zero, %zero, %zero) to (%three, %four, %two) step (%one, %one, %one) {
      %old = memref.load %counts[%i, %j, %k] : memref<3x4x2xi32>
      %next = arith.addi %old, %increment : i32
      memref.store %next, %counts[%i, %j, %k] : memref<3x4x2xi32>
      scf.reduce
    }
    return
  }
  func.func @entry() {
    %null = llvm.mlir.zero : !llvm.ptr
    "muon.launch"(%null) {kernel = @kernel, warps_per_core = 1 : i32} : (!llvm.ptr) -> ()
    return
  }
  func.func @get_count(%i: i32, %j: i32, %k: i32) -> i32 {
    %ii = arith.index_cast %i : i32 to index
    %jj = arith.index_cast %j : i32 to index
    %kk = arith.index_cast %k : i32 to index
    %counts = memref.get_global @counts : memref<3x4x2xi32>
    %value = memref.load %counts[%ii, %jj, %kk] : memref<3x4x2xi32>
    return %value : i32
  }
}
