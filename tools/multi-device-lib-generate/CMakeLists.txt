add_llvm_executable(multi-device-lib-generate multi-device-lib-generate.cpp)

target_link_libraries(multi-device-lib-generate PUBLIC
MultiDeviceCompileUtils
TPUMLIRTpu TPUMLIRTop
MLIRIR MLIRParser MLIRPass MLIRFuncDialect)