include "mlir/IR/OpBase.td"

def Muon_Dialect : Dialect {
  let name = "muon";
  let cppNamespace = "::mlir::muon";
  let summary = "Muon SIMT execution and synchronization";
}

class Muon_Op<string mnemonic> : Op<Muon_Dialect, mnemonic> {}

def Muon_LaunchOp : Muon_Op<"launch"> {
  let summary = "Launch one Muon callback on the selected cluster";
  let arguments = (ins AnyType:$arg, FlatSymbolRefAttr:$kernel,
                       I32Attr:$warps_per_core);
  let hasVerifier = 1;
}

def Muon_ThreadIdOp : Muon_Op<"thread_id"> {
  let summary = "Lane/thread ID within the scheduled thread block";
  let results = (outs I32:$id);
}

def Muon_BlockIdOp : Muon_Op<"block_id"> {
  let summary = "Scheduled thread-block ID";
  let results = (outs I32:$id);
}

def Muon_ThreadsPerBlockOp : Muon_Op<"threads_per_block"> {
  let summary = "Scheduled threads per thread block";
  let results = (outs I32:$count);
}

def Muon_BarrierOp : Muon_Op<"barrier"> {
  let summary = "Convergent all-participant warp barrier";
  let arguments = (ins I32Attr:$barrier_id, I32Attr:$num_warps);
  let hasVerifier = 1;
}

def Muon_FenceOp : Muon_Op<"fence"> {
  let summary = "Drain Muon global-memory operations";
}

def Muon_SmemFenceOp : Muon_Op<"smem_fence"> {
  let summary = "Order Muon shared-memory operations";
}
