# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized")
file(GLOB _sources *.cpp Dnnl/*.cpp Helper/*.cpp)

add_library(TPUMLIRSupport STATIC ${_sources})
target_link_libraries(TPUMLIRSupport PUBLIC
MLIRIR
MLIRSupport
MLIRQuantDialect
MLIRQuantUtils
MLIRInferenceInterface
cnpy
dnnl
z)
add_dependencies(TPUMLIRSupport ModuleAttrGen MLIRTopOpsIncGen)
