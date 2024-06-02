if(ONLY_BUILD_PLAIN_RUNTIME)
    add_subdirectory(Runtime)    
else()

add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Pass)
add_subdirectory(Pipelines)
add_subdirectory(Utils)
add_subdirectory(Runtime)
add_subdirectory(Kernels)

add_library(MultiDeviceInitUtils STATIC InitUtils.cpp)
target_link_libraries(MultiDeviceInitUtils PUBLIC OMONNXRewrite OMShapeInferencePass 
OMInstrumentONNX OMHybridTransform OMONNXSimplifyShapeRelatedOps OMONNXToTOSA
MultiDevicePipelines MultiDevicePasses MultiDeviceConvertPasses
)

add_library(MultiDeviceInitTranslations STATIC InitTranslations.cpp)
target_link_libraries(MultiDeviceInitTranslations PUBLIC
MLIRToLLVMIRTranslationRegistration
MultiDeviceDeviceDialect
)

endif()