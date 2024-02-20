add_library(ConvertONNXToTosaFix STATIC 
ConvertONNXToTosa.cpp
Tensor/return.cpp
Tensor/flatten.cpp
nn/matmul.cpp
Reduce/ReduceMean.cpp)
add_dependencies(ConvertONNXToTosaFix MultiDeviceConversionPassIncGen)
target_link_libraries(ConvertONNXToTosaFix PUBLIC   
OMONNXOps
MLIRTosaDialect
MLIRTransforms)