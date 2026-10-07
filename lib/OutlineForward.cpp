#include "Muon/Passes.h"
#include "Muon/MuonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
namespace {
// The symbols are a storage binding supplied by the caller.  The computation
// itself is cloned from the bufferized model2MLIR function, without rebuilding
// its arithmetic or loop body.
class OutlineForwardPass
    : public PassWrapper<OutlineForwardPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineForwardPass)
  OutlineForwardPass() = default;
  OutlineForwardPass(const OutlineForwardPass &other) : PassWrapper(other) {
    inputs = other.inputs;
    output = other.output;
    warps = other.warps;
    sharedScratch = other.sharedScratch;
  }
  Option<std::string> inputs{
      *this, "inputs", llvm::cl::desc("Comma-separated external input symbols")};
  Option<std::string> output{
      *this, "output", llvm::cl::desc("External result buffer symbol")};
  Option<unsigned> warps{
      *this, "warps", llvm::cl::desc("Warps per Muon core"),
      llvm::cl::init(4)};
  Option<bool> sharedScratch{
      *this, "shared-scratch",
      llvm::cl::desc("Hoist intermediate static allocations to globals and synchronize stages"),
      llvm::cl::init(false)};

  StringRef getArgument() const final { return "outline-forward-to-muon"; }
  StringRef getDescription() const final {
    return "Bind a bufferized model2MLIR forward to Muon storage and callback ABI";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect, muon::MuonDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto forward = module.lookupSymbol<func::FuncOp>("forward");
    if (!forward || forward.isExternal() || !forward.getBody().hasOneBlock() ||
        forward.getNumResults() != 1 || warps < 1 || warps > 8 ||
        module.lookupSymbol("muon_kernel") || module.lookupSymbol("entry")) {
      module.emitError("expected one defined forward with one result, warps in [1,8], and free muon_kernel/entry symbols");
      signalPassFailure();
      return;
    }
    SmallVector<StringRef> names;
    llvm::SplitString(inputs.getValue(), names, ",");
    if (names.size() != forward.getNumArguments() || output.getValue().empty()) {
      forward.emitError("input symbol count must equal forward argument count; output is required");
      signalPassFailure();
      return;
    }
    for (StringRef name : names)
      if (name.empty()) {
        forward.emitError("input symbols must be nonempty");
        signalPassFailure();
        return;
      }
    auto ret = dyn_cast<func::ReturnOp>(forward.getBody().front().getTerminator());
    if (!ret || ret.getNumOperands() != 1 ||
        !isa<MemRefType>(forward.getResultTypes().front())) {
      forward.emitError("forward must return one ranked memref");
      signalPassFailure();
      return;
    }
    auto alloc = ret.getOperand(0).getDefiningOp<memref::AllocOp>();
    auto copiedArg = dyn_cast<BlockArgument>(ret.getOperand(0));
    bool copyIdentity = !alloc && copiedArg &&
                        copiedArg.getOwner() == &forward.getBody().front();
    if (!copyIdentity && (!alloc || alloc.getNumOperands())) {
      forward.emitError("result must be a direct memref.alloc or input identity");
      signalPassFailure();
      return;
    }
    if (copyIdentity &&
        (forward.getBody().front().getOperations().size() != 1 ||
         cast<MemRefType>(copiedArg.getType()).getRank() != 1)) {
      forward.emitError("input identity copy requires a single return and rank-one buffer");
      signalPassFailure();
      return;
    }
    auto returnedType = cast<MemRefType>(ret.getOperand(0).getType());
    auto outputType = copyIdentity
                          ? MemRefType::get(returnedType.getShape(),
                                            returnedType.getElementType(), AffineMap(),
                                            returnedType.getMemorySpace())
                          : cast<MemRefType>(alloc.getType());
    if (!outputType.hasStaticShape() || !outputType.getLayout().isIdentity()) {
      forward.emitError("output allocation must have static shape and identity layout");
      signalPassFailure();
      return;
    }
    SmallVector<memref::AllocOp> scratch;
    bool hasDealloc = false;
    forward.walk([&](memref::DeallocOp) { hasDealloc = true; });
    if (hasDealloc) {
      forward.emitError("outlined callback cannot deallocate bound storage");
      signalPassFailure();
      return;
    }
    for (Operation &op : forward.getBody().front()) {
      auto candidate = dyn_cast<memref::AllocOp>(&op);
      if (!candidate || candidate == alloc)
        continue;
      auto type = cast<MemRefType>(candidate.getType());
      if (!sharedScratch || candidate.getNumOperands() ||
          !type.hasStaticShape() || !type.getLayout().isIdentity()) {
        forward.emitError("intermediate allocation requires --shared-scratch and static identity layout");
        signalPassFailure();
        return;
      }
      scratch.push_back(candidate);
    }
    SmallVector<MemRefType> boundTypes;
    for (Type type : forward.getArgumentTypes()) {
      auto memrefType = dyn_cast<MemRefType>(type);
      if (!memrefType || !memrefType.hasStaticShape()) {
        forward.emitError("all inputs must be statically shaped ranked memrefs");
        signalPassFailure();
        return;
      }
      boundTypes.push_back(MemRefType::get(memrefType.getShape(),
                                           memrefType.getElementType(), AffineMap(),
                                           memrefType.getMemorySpace()));
      if (!memref::CastOp::areCastCompatible(TypeRange{boundTypes.back()},
                                            TypeRange{type})) {
        forward.emitError("input layout cannot be bound to a dense external buffer");
        signalPassFailure();
        return;
      }
    }
    SmallVector<std::pair<StringRef, MemRefType>> bindings;
    for (auto [name, type] : llvm::zip(names, boundTypes))
      bindings.emplace_back(name, type);
    bindings.emplace_back(output.getValue(), outputType);
    for (auto [index, binding] : llvm::enumerate(bindings)) {
      auto [name, type] = binding;
      for (unsigned prior = 0; prior < index; ++prior)
        if (bindings[prior].first == name && bindings[prior].second != type) {
          forward.emitError("one storage symbol has incompatible buffer types: ")
              << name;
          signalPassFailure();
          return;
        }
      if (auto symbol = module.lookupSymbol(name)) {
        auto global = dyn_cast<memref::GlobalOp>(symbol);
        if (!global || global.getType() != type || global.getInitialValue()) {
          forward.emitError("storage symbol collision or incompatible global: ")
              << name;
          signalPassFailure();
          return;
        }
      }
    }
    for (auto [index, allocation] : llvm::enumerate(scratch)) {
      std::string name = "__muon_scratch_" + std::to_string(index);
      if (module.lookupSymbol(name)) {
        forward.emitError("reserved Muon scratch symbol already exists: ") << name;
        signalPassFailure();
        return;
      }
    }

    IRRewriter writer(module.getContext());
    Location loc = forward.getLoc();
    writer.setInsertionPointToStart(module.getBody());
    for (auto [name, type] : bindings) {
      if (!module.lookupSymbol(name))
        writer.create<memref::GlobalOp>(loc, name, StringAttr(), type,
                                        Attribute(), false, IntegerAttr());
    }
    for (auto [index, allocation] : llvm::enumerate(scratch)) {
      std::string name = "__muon_scratch_" + std::to_string(index);
      writer.create<memref::GlobalOp>(loc, name, writer.getStringAttr("private"),
                                      cast<MemRefType>(allocation.getType()),
                                      UnitAttr::get(module.getContext()), false,
                                      IntegerAttr());
    }
    Type ptr = LLVM::LLVMPointerType::get(module.getContext());
    Type i32 = writer.getI32Type();
    auto callbackType = writer.getFunctionType({ptr, i32, i32, i32}, {});
    writer.setInsertionPointAfter(forward);
    auto callback = writer.create<func::FuncOp>(loc, "muon_kernel", callbackType);
    Block *body = callback.addEntryBlock();
    writer.setInsertionPointToStart(body);
    IRMapping mapping;
    for (auto [arg, name, type] : llvm::zip(forward.getArguments(), names,
                                             boundTypes)) {
      Value bound = writer.create<memref::GetGlobalOp>(loc, type, name);
      if (type != arg.getType())
        bound = writer.create<memref::CastOp>(loc, arg.getType(), bound);
      mapping.map(arg, bound);
    }
    Value result = writer.create<memref::GetGlobalOp>(loc, outputType,
                                                       output.getValue());
    if (alloc)
      mapping.map(alloc.getResult(), result);
    for (auto [index, allocation] : llvm::enumerate(scratch)) {
      std::string name = "__muon_scratch_" + std::to_string(index);
      Value global = writer.create<memref::GetGlobalOp>(
          loc, allocation.getType(), name);
      mapping.map(allocation.getResult(), global);
    }
    for (Operation &op : forward.getBody().front().without_terminator()) {
      if (isa<memref::AllocOp>(op))
        continue;
      writer.clone(op, mapping);
      if (sharedScratch && isa<scf::ParallelOp>(op)) {
        writer.create<muon::FenceOp>(loc);
        writer.create<muon::BarrierOp>(loc, 0, warps);
      }
    }
    if (copyIdentity) {
      Value source = mapping.lookup(copiedArg);
      Value zero = writer.create<arith::ConstantIndexOp>(loc, 0);
      Value upper = writer.create<arith::ConstantIndexOp>(loc,
                                                          outputType.getDimSize(0));
      Value one = writer.create<arith::ConstantIndexOp>(loc, 1);
      writer.create<scf::ParallelOp>(
          loc, ValueRange{zero}, ValueRange{upper}, ValueRange{one},
          [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs) {
            Value value = builder.create<memref::LoadOp>(bodyLoc, source, ivs);
            builder.create<memref::StoreOp>(bodyLoc, value, result, ivs);
          });
    }
    writer.create<func::ReturnOp>(loc);

    writer.setInsertionPointAfter(callback);
    auto entry = writer.create<func::FuncOp>(loc, "entry",
                                               writer.getFunctionType({}, {}));
    Block *entryBody = entry.addEntryBlock();
    writer.setInsertionPointToStart(entryBody);
    Value null = writer.create<LLVM::ZeroOp>(loc, ptr);
    writer.create<muon::LaunchOp>(loc, null, "muon_kernel", warps);
    writer.create<muon::FenceOp>(loc);
    writer.create<func::ReturnOp>(loc);
    writer.eraseOp(forward);
  }
};
} // namespace

namespace mlir::muon {
void registerOutlineForwardPass() { PassRegistration<OutlineForwardPass>(); }
} // namespace mlir::muon
