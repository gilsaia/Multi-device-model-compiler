string(REPLACE "$<CONFIG>" ${CMAKE_BUILD_TYPE} MULTI_DEVICE_TOOLS_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(TPU_MLIR_TOOLCHAIN_DIR ${CMAKE_SOURCE_DIR}/3rdparty/tpu-mlir/third_party/nntoolchain/lib)
set(TPUC_ROOT_DIR ${CMAKE_SOURCE_DIR}/3rdparty/tpu-mlir/third_party/nntoolchain)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MULTI_DEVICE_TEST_DEPENDS
FileCheck
count
not
multi-device-opt
multi-device-translate
multi-device-lib-generate
multi-device-cpu-runner
model-converter
)

if(USE_CUDA_SUPPORT)
list(APPEND MULTI_DEVICE_TEST_DEPENDS multi-device-gpu-runner)
endif()

add_lit_testsuite(check-multi-device "Running the multi-device tests..."
${CMAKE_CURRENT_BINARY_DIR}
DEPENDS ${MULTI_DEVICE_TEST_DEPENDS}
)
set_target_properties(check-multi-device PROPERTIES FOLDER "Tests")

add_lit_testsuites(MULTI-DEVICE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MULTI_DEVICE_TEST_DEPENDS})