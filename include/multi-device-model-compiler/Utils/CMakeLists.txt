configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ExternalUtil.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.h.cfg
    @ONLY
    )

file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/ExternalUtil.h
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.h.cfg)

# CMAKE_CFG_INTDIR is . for single-config generators such as make, and
# it has a value (e.g. $(Configuration)) otherwise, so we can use it to
# determine whether we are dealing with a multi-config generator.
if (NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
  set(FILE_GENERATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
else()
  set(FILE_GENERATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif()

set(MultiDeviceConfigDir ${FILE_GENERATE_DIR} CACHE PATH "Dir of config")

add_custom_target(MultiDeviceExternalUtil DEPENDS ${FILE_GENERATE_DIR}/ExternalUtil.h)