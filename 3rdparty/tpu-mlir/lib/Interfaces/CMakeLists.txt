set(
  LLVM_OPTIONAL_SOURCES
  InferenceInterface.cpp
  WeightReorderInterface.cpp
  GlobalGenInterface.cpp
  FlopsInterface.cpp
  LocalGenInterface.cpp
  DynLocalGenInterface.cpp
  DynGlobalGenInterface.cpp
  TypeInterface.cpp
  ShapeInterface.cpp
  IndexingMapsInterface.cpp
)

function(add_mlir_interface_library name)
  add_library(MLIR${name} STATIC ${name}.cpp)
  target_include_directories(MLIR${name} PUBLIC ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces)
  add_dependencies(MLIR${name} MLIR${name}IncGen ModuleAttrGen)
  target_link_libraries(MLIR${name} PUBLIC MLIRIR)
endfunction(add_mlir_interface_library)

add_mlir_interface_library(InferenceInterface)
add_mlir_interface_library(GlobalGenInterface)
add_mlir_interface_library(FlopsInterface)
add_mlir_interface_library(LocalGenInterface)
add_mlir_interface_library(DynLocalGenInterface)
add_mlir_interface_library(DynGlobalGenInterface)
add_mlir_interface_library(TypeInterface)
add_mlir_interface_library(ShapeInterface)
add_mlir_interface_library(IndexingMapsInterface)
