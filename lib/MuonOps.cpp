#include "Muon/MuonOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
using namespace mlir;
using namespace mlir::muon;

LogicalResult LaunchOp::verify() {
  if (!mlir::isa<LLVM::LLVMPointerType>(getArg().getType()))
    return emitOpError("arg must be !llvm.ptr");
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) return emitOpError("requires a module");
  auto kernel = module.lookupSymbol<func::FuncOp>(getKernelAttr().getValue());
  if (!kernel) return emitOpError("references a missing func.func kernel");
  if (getWarpsPerCore() < 1 || getWarpsPerCore() > 8)
    return emitOpError("warps_per_core must be in [1, 8]");
  auto type = kernel.getFunctionType();
  if (type.getNumInputs() != 4 ||
      !mlir::isa<LLVM::LLVMPointerType>(type.getInput(0)) ||
      !type.getInput(1).isInteger(32) ||
      !type.getInput(2).isInteger(32) || !type.getInput(3).isInteger(32) ||
      type.getNumResults() != 0)
    return emitOpError("kernel must have callback ABI (opaque_arg, i32 tid, i32 tpb, i32 block_id) -> ()");
  return success();
}

LogicalResult BarrierOp::verify() {
  if (getBarrierId() < 0 || getBarrierId() > 15)
    return emitOpError("barrier_id must be in [0, 15]");
  if (getNumWarps() < 1 || getNumWarps() > 16)
    return emitOpError("num_warps must be in [1, 16]");
  return success();
}

#define GET_OP_CLASSES
#include "Muon/MuonOps.cpp.inc"
