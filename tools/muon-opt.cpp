#include "Muon/MuonDialect.h"
#include "Muon/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  mlir::muon::registerLowerRuntimePass();
  mlir::muon::registerDistributeParallelPass();
  mlir::muon::registerOutlineForwardPass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::muon::MuonDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect, mlir::cf::ControlFlowDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect, mlir::vector::VectorDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv,
                                                  "Muon SIMT dialect\n", registry));
}
