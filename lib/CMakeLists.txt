add_mlir_dialect_library(MLIRMuon
  MuonDialect.cpp
  MuonOps.cpp
  LowerRuntime.cpp
  DistributeParallel.cpp
  OutlineForward.cpp
  DEPENDS MLIRMuonOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRPass MLIRFuncDialect MLIRLLVMDialect
    MLIRArithDialect MLIRSCFDialect MLIRMemRefDialect)
