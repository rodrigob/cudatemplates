project(cuda_bugs)
cmake_minimum_required(VERSION 2.6)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

# Cuda:
set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
find_package(CUDA REQUIRED)
find_package(CudaTemplates REQUIRED)
add_definitions(-DUSE_CUDA)
include_directories(
  ${CUDATEMPLATES_INCLUDE_DIR}
  ${FOUND_CUDA_NVCC_INCLUDE}
  ${CUDA_CUT_INCLUDE_DIR}
)

cuda_include_directories(
  ${CUDA_CUT_INCLUDE_DIR}
)

find_package(Boost 1.35 REQUIRED)
include_directories(
  ${Boost_INCLUDE_DIR}
)

#cuda_add_executable(bug1 bug1.cu)
#cuda_add_executable(bug2 bug2.cpp)
#cuda_add_executable(float3 float3.cpp)
#cuda_add_executable(bug3 bug3.cu)
#cuda_add_executable(bug4 bug4.cu)
#cuda_add_executable(bug5 bug5.cu)
#cuda_add_executable(bug7 bug7.cu)
#cuda_add_executable(bug8 bug8.cpp)
cuda_add_executable(bug9 bug9.cpp)
