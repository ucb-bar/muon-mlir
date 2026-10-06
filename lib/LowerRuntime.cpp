#include "Muon/Passes.h"
#include "Muon/MuonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace {
class LowerRuntimePass : public PassWrapper<LowerRuntimePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRuntimePass)
  StringRef getArgument() const final { return "lower-muon-runtime"; }
  StringRef getDescription() const final { return "Lower Muon control ops to the Muon runtime ABI"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    Type i32 = IntegerType::get(ctx, 32);
    Type ptr = LLVM::LLVMPointerType::get(ctx);
    IRRewriter writer(ctx);
    auto declare = [&](StringRef name, FunctionType type) {
      if (module.lookupSymbol<func::FuncOp>(name)) return;
      writer.setInsertionPointToStart(module.getBody());
      auto fn = writer.create<func::FuncOp>(module.getLoc(), name, type);
      fn.setPrivate();
    };
    SmallVector<Operation *> work;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "muon") work.push_back(op);
    });
    for (Operation *op : work) {
      writer.setInsertionPoint(op);
      Location loc = op->getLoc();
      if (auto id = dyn_cast<muon::ThreadIdOp>(op)) {
        auto fn = op->getParentOfType<func::FuncOp>();
        if (!fn || fn.getNumArguments() != 4) {
          op->emitError("thread_id requires a Muon callback");
          signalPassFailure(); return;
        }
        writer.replaceOp(op, fn.getArgument(1));
      } else if (auto id = dyn_cast<muon::BlockIdOp>(op)) {
        auto fn = op->getParentOfType<func::FuncOp>();
        if (!fn || fn.getNumArguments() != 4) {
          op->emitError("block_id requires a Muon callback");
          signalPassFailure(); return;
        }
        writer.replaceOp(op, fn.getArgument(3));
      } else if (auto count = dyn_cast<muon::ThreadsPerBlockOp>(op)) {
        auto fn = op->getParentOfType<func::FuncOp>();
        if (!fn || fn.getNumArguments() != 4) {
          op->emitError("threads_per_block requires a Muon callback");
          signalPassFailure(); return;
        }
        writer.replaceOp(op, fn.getArgument(2));
      } else if (auto barrier = dyn_cast<muon::BarrierOp>(op)) {
        declare("muon_mlir_barrier", FunctionType::get(ctx, {i32, i32}, {}));
        writer.setInsertionPoint(op);
        Value id = writer.create<arith::ConstantIntOp>(loc, barrier.getBarrierId(), 32);
        Value warps = writer.create<arith::ConstantIntOp>(loc, barrier.getNumWarps(), 32);
        writer.create<func::CallOp>(loc, "muon_mlir_barrier", TypeRange{}, ValueRange{id, warps});
        writer.eraseOp(op);
      } else if (isa<muon::FenceOp>(op) || isa<muon::SmemFenceOp>(op)) {
        StringRef name = isa<muon::FenceOp>(op) ? "muon_mlir_fence" : "muon_mlir_smem_fence";
        declare(name, FunctionType::get(ctx, {}, {}));
        writer.setInsertionPoint(op);
        writer.create<func::CallOp>(loc, name, TypeRange{}, ValueRange{});
        writer.eraseOp(op);
      } else if (auto launch = dyn_cast<muon::LaunchOp>(op)) {
        auto kernel = module.lookupSymbol<func::FuncOp>(launch.getKernelAttr().getValue());
        if (!kernel) { op->emitError("missing callback"); signalPassFailure(); return; }
        declare("mu_schedule", FunctionType::get(ctx, {kernel.getFunctionType(), ptr, i32}, {}));
        writer.setInsertionPoint(op);
        Value callback = writer.create<func::ConstantOp>(loc, kernel.getFunctionType(),
            FlatSymbolRefAttr::get(ctx, kernel.getSymName()));
        Value occupancy = writer.create<arith::ConstantIntOp>(loc, launch.getWarpsPerCore(), 32);
        writer.create<func::CallOp>(loc, "mu_schedule", TypeRange{},
                                    ValueRange{callback, launch.getArg(), occupancy});
        writer.eraseOp(op);
      }
    }
  }
};
} // namespace
namespace mlir::muon {
void registerLowerRuntimePass() { PassRegistration<LowerRuntimePass>(); }
}
