add_library(ConvertDeviceToTPU STATIC ConvertDeviceToTPU.cpp
Matmul.cpp
Clip.cpp
Conv2d.cpp
Pool2d.cpp)
add_dependencies(ConvertDeviceToTPU MultiDeviceConversionPassIncGen)
target_link_libraries(ConvertDeviceToTPU PUBLIC
MLIRTosaDialect MLIRTransforms MLIRIR MLIRFuncDialect
TPUMLIRTpu TPUMLIRTop TPUMLIRSupport)
