#include "Muon/MuonDialect.h"
#include "Muon/MuonOps.h"
using namespace mlir;
using namespace mlir::muon;
#include "Muon/MuonOpsDialect.cpp.inc"
void MuonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Muon/MuonOps.cpp.inc"
      >();
}
