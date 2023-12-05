add_llvm_executable(model-converter model-converter.cpp)

target_link_libraries(model-converter PRIVATE MLIROptLib OMBuilder OMCompilerUtils)