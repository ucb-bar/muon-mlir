add_mlir_dialect_library(MLIRMuon
  MuonDialect.cpp
  MuonOps.cpp
  LowerRuntime.cpp
  DEPENDS MLIRMuonOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRPass MLIRFuncDialect MLIRLLVMDialect MLIRArithDialect)
