file(GLOB _sources
          IR/*.cpp
          Interfaces/*.cpp
          Canonicalize/*.cpp
          Transforms/*.cpp
          Transforms/ChipOptimize/*.cpp
    )

add_library(TPUMLIRTop STATIC ${_sources})
add_dependencies(TPUMLIRTop
MLIRTopAttrGen
MLIRTopOpsIncGen
MLIRTopPassIncGen
MLIRInferenceInterfaceIncGen
MLIRFlopsInterfaceIncGen
MLIRShapeInterfaceIncGen)
target_include_directories(TPUMLIRTop PUBLIC
${CMAKE_SOURCE_DIR}/3rdparty/tpu-mlir/include/tpu_mlir/Dialect/Top)
target_link_libraries(TPUMLIRTop PUBLIC
MLIRIR
MLIRPass
MLIRSupport
MLIRParser
MLIRFuncDialect
MLIRQuantDialect
MLIRRewrite
MLIRTransforms
MLIRInferenceInterface
MLIRFlopsInterface
MLIRShapeInterface
TPUMLIRSupport
TPUMLIRTrait)
