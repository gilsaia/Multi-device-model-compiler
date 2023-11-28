add_llvm_executable(multi-device-opt multi-device-opt.cpp)

target_link_libraries(multi-device-opt PRIVATE MLIROptLib)