project(cuda_bugs)
cmake_minimum_required(VERSION 2.6)

# Cuda:
SET(CUDA_BUILD_TYPE Device CACHE STRING "Choose 'Device' or 'Emulation'")
  FIND_PACKAGE(Cuda REQUIRED)
  FIND_PACKAGE(CudaTemplates REQUIRED)
  ADD_DEFINITIONS(-DUSE_CUDA)
  INCLUDE_DIRECTORIES(
    ${CUDATEMPLATES_INCLUDE_DIR}
    ${FOUND_CUDA_NVCC_INCLUDE}
  )

FIND_PACKAGE(Boost 1.35 REQUIRED)
INCLUDE_DIRECTORIES(
  ${Boost_INCLUDE_DIR}
)

#cuda_add_executable(bug1 bug1.cu)

CUDA_ADD_EXECUTABLE(bug2 bug2.cpp)