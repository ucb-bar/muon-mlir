#include "Muon/Passes.h"
#include "Muon/MuonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

using namespace mlir;
namespace {
class DistributeParallelPass
    : public PassWrapper<DistributeParallelPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributeParallelPass)
  DistributeParallelPass() = default;
  DistributeParallelPass(const DistributeParallelPass &other)
      : PassWrapper(other) { blocks = other.blocks; }
  Option<unsigned> blocks{*this, "blocks",
                          llvm::cl::desc("Number of callback block IDs in the target launch"),
                          llvm::cl::init(1)};
  StringRef getArgument() const final { return "distribute-scf-parallel-to-muon"; }
  StringRef getDescription() const final {
    return "Map callback-local reduction-free parallel loops across Muon lanes";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    if (blocks == 0) {
      getOperation().emitError("Muon block count must be positive");
      signalPassFailure();
      return;
    }
    ModuleOp module = getOperation();
    llvm::SmallPtrSet<Operation *, 8> callbacks;
    module.walk([&](muon::LaunchOp launch) {
      if (auto fn = module.lookupSymbol<func::FuncOp>(
              launch.getKernelAttr().getValue()))
        callbacks.insert(fn.getOperation());
    });
    IRRewriter writer(module.getContext());
    for (Operation *operation : callbacks) {
      auto callback = cast<func::FuncOp>(operation);
      if (callback.getNumArguments() != 4 ||
          !callback.getArgument(1).getType().isInteger(32) ||
          !callback.getArgument(2).getType().isInteger(32) ||
          !callback.getArgument(3).getType().isInteger(32)) {
        callback.emitError("Muon callback must carry tid, threads, and block i32 arguments");
        signalPassFailure();
        return;
      }
      SmallVector<scf::ParallelOp> loops;
      callback.walk([&](scf::ParallelOp loop) { loops.push_back(loop); });
      for (scf::ParallelOp parallel : loops) {
        if (parallel.getNumReductions() != 0 || parallel.getNumLoops() == 0 ||
            parallel->getParentOfType<scf::ParallelOp>()) {
          parallel.emitError("only unnested reduction-free parallel loops are supported");
          signalPassFailure();
          return;
        }
        Location loc = parallel.getLoc();
        writer.setInsertionPoint(parallel);
        Type index = writer.getIndexType();
        Value tid = writer.create<arith::IndexCastOp>(
            loc, index, callback.getArgument(1));
        Value threads = writer.create<arith::IndexCastOp>(
            loc, index, callback.getArgument(2));
        Value block = writer.create<arith::IndexCastOp>(
            loc, index, callback.getArgument(3));
        Value blockBase = writer.create<arith::MulIOp>(loc, block, threads);
        Value lane = writer.create<arith::AddIOp>(loc, blockBase, tid);
        Value blockCount = writer.create<arith::ConstantIndexOp>(loc, blocks);
        Value totalThreads = writer.create<arith::MulIOp>(loc, threads, blockCount);
        Value laneOffset = writer.create<arith::MulIOp>(
            loc, lane, parallel.getStep().front());
        Value first = writer.create<arith::AddIOp>(
            loc, parallel.getLowerBound().front(), laneOffset);
        Value stride = writer.create<arith::MulIOp>(
            loc, totalThreads, parallel.getStep().front());
        IRMapping mapping;
        SmallVector<Value> ivs = parallel.getInductionVars();
        std::function<void(unsigned)> emit = [&](unsigned dim) {
          if (dim == parallel.getNumLoops()) {
            for (Operation &op : parallel.getBody()->without_terminator())
              writer.clone(op, mapping);
            return;
          }
          Value lower = dim == 0 ? first : parallel.getLowerBound()[dim];
          Value step = dim == 0 ? stride : parallel.getStep()[dim];
          auto loop = writer.create<scf::ForOp>(
              loc, lower, parallel.getUpperBound()[dim], step);
          mapping.map(ivs[dim], loop.getInductionVar());
          writer.setInsertionPointToStart(loop.getBody());
          emit(dim + 1);
          writer.setInsertionPointAfter(loop);
        };
        emit(0);
        writer.eraseOp(parallel);
      }
    }
  }
};
} // namespace

namespace mlir::muon {
void registerDistributeParallelPass() {
  PassRegistration<DistributeParallelPass>();
}
} // namespace mlir::muon
