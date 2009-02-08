find_package(Cuda REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)
cuda_include_directories(${CMAKE_SOURCE_DIR}/include)

add_custom_command(
  OUTPUT auto/convert_type.cu
  DEPENDS auto/convert_type.cmake auto/convert_type_begin.cu auto/convert_type_template.cu
  COMMAND ${CMAKE_COMMAND} -P convert_type.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/auto
)

cuda_add_library(cudatemplates SHARED auto/convert_type.cu)
