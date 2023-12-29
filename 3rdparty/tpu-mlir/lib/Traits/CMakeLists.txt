file(GLOB _sources *.cpp)

add_library(TPUMLIRTrait STATIC ${_sources})
add_dependencies(TPUMLIRTrait ModuleAttrGen)
target_link_libraries(TPUMLIRTrait PUBLIC
MLIRIR
MLIRQuantDialect)

# add_llvm_library(TPUMLIRTrait
#   ${_sources}
#   LINK_LIBS PUBLIC
#   MLIRIR
#   MLIRQuantDialect
# )