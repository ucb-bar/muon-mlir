#include "Muon/MuonDialect.h"
#include "Muon/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  mlir::muon::registerLowerRuntimePass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::muon::MuonDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv,
                                                  "Muon SIMT dialect\n", registry));
}
