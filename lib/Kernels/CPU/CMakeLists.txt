add_library(MultiDeviceCPUOps SHARED 
Ops.cpp
LLM.cpp)
target_include_directories(MultiDeviceCPUOps PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/xxHash)
target_link_libraries(MultiDeviceCPUOps PRIVATE
xxHash::xxhash)
target_link_libraries(MultiDeviceCPUOps PUBLIC
LLVMSupport
dnnl)