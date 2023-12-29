include_directories(include)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/tpu-mlir/include)

#-------------------------------------------------------------------------------
# cnpy Configuration
#-------------------------------------------------------------------------------
set(CNPY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy)
include_directories(${CNPY_PATH})
#-------------------------------------------------------------------------------
# nntoolchain Configuration
#-------------------------------------------------------------------------------
set(NNTOOLCHAIN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nntoolchain)
include_directories(${NNTOOLCHAIN_PATH}/include)
link_directories(${NNTOOLCHAIN_PATH}/lib)
#-------------------------------------------------------------------------------
# CV18xx Configuration
#-------------------------------------------------------------------------------
set(CV18XX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/CV18xx)
include_directories(${CV18XX_PATH}/include)
link_directories(${CV18XX_PATH}/lib)
#-------------------------------------------------------------------------------
# PROGRESSBAR Configuration
#-------------------------------------------------------------------------------
set(PROGRESSBAR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/progressbar)
include_directories(${PROGRESSBAR_PATH}/include)

set(TPU_MLIR_THIRDPARTY_INCLUDE 
${CNPY_PATH} 
${NNTOOLCHAIN_PATH}/include 
${CV18XX_PATH}/include 
${PROGRESSBAR_PATH}/include 
PARENT_SCOPE)
set(TPU_MLIR_THIRDPARTY_LIB
${NNTOOLCHAIN_PATH}/lib
${CV18XX_PATH}/lib
PARENT_SCOPE)

add_subdirectory(include)
add_subdirectory(third_party)
add_subdirectory(lib)
