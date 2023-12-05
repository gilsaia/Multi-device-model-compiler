add_llvm_executable(multi-device-opt multi-device-opt.cpp)

target_include_directories(multi-device-opt PRIVATE ${CMAKE_BINARY_DIR}/3rdparty/onnx-mlir)
target_link_libraries(multi-device-opt PRIVATE MLIROptLib PassUtils OMONNXOps)