add_executable(muon-opt muon-opt.cpp)
target_link_libraries(muon-opt PRIVATE MLIRMuon MLIROptLib MLIRIR MLIRFuncDialect MLIRLLVMDialect MLIRArithDialect MLIRMemRefDialect MLIRSCFDialect)
mlir_check_all_link_libraries(muon-opt)
