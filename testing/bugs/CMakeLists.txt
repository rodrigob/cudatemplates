project(cuda_bugs)
cmake_minimum_required(VERSION 2.6)

# Cuda:
set(CUDA_BUILD_CUBIN OFF)
set(CUDA_BUILD_TYPE Device)
#set(CUDA_BUILD_TYPE Emulation)
find_package(Cuda REQUIRED)
include_directories(${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE})

cuda_add_executable(bug1 bug1.cu)
