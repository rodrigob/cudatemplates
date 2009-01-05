# Boost:
find_package(Boost)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIR})
endif(Boost_FOUND)

# Cuda:
set(CUDA_BUILD_CUBIN OFF)
set(CUDA_BUILD_TYPE Device)
#set(CUDA_BUILD_TYPE Emulation)
find_package(Cuda REQUIRED)
include_directories(${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE})

# cudatemplates
find_package(CudaTemplates REQUIRED)
include_directories(${CUDATEMPLATES_INCLUDE_DIR})
cuda_include_directories(${CUDATEMPLATES_INCLUDE_DIR})

# ITK:
find_package(ITK)
if(ITK_FOUND)
  include(${ITK_USE_FILE})
endif(ITK_FOUND)

# OpenCV:
find_package(OpenCV)
include_directories(${OPENCV_INCLUDE_DIR})

# OpenGL:
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
add_definitions(-DGL_GLEXT_PROTOTYPES)

# GLUT:
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})

# PNG:
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})


include_directories(${CMAKE_SOURCE_DIR}/include)
cuda_include_directories(${CMAKE_SOURCE_DIR}/include)

#add_definitions(-Wall)

# test programs:

add_executable(copy copy.cpp)
target_link_libraries(copy ${CUDA_TARGET_LINK})

add_executable(demo demo.cpp)
target_link_libraries(demo ${CUDA_TARGET_LINK})

add_executable(fft fft.cpp)
target_link_libraries(fft ${CUDA_TARGET_LINK} ${CUFFT_TARGET_LINK})

add_executable(blas blas.cpp)
target_link_libraries(blas ${CUDA_TARGET_LINK} ${CUBLAS_TARGET_LINK})

if(ITK_FOUND)
  add_executable(itk itk.cpp)
  target_link_libraries(itk ${CUDA_TARGET_LINK} ${ITK_LIBRARIES})
endif(ITK_FOUND)

if(Boost_FOUND)
  add_executable(gil gil.cpp)
  target_link_libraries(gil ${CUDA_TARGET_LINK} ${PNG_LIBRARIES})
endif(Boost_FOUND)

if(OpenCV_FOUND)
  add_executable(ipl ipl.cpp)
  target_link_libraries(ipl ${CUDA_TARGET_LINK} ${OPENCV_LIBRARIES})
endif(OpenCV_FOUND)

add_executable(array array.cpp)
target_link_libraries(array ${CUDA_TARGET_LINK})

cuda_add_executable(kernel kernel.cu)
target_link_libraries(kernel ${CUDA_TARGET_LINK})

add_executable(texture texture.cu)
target_link_libraries(texture ${CUDA_TARGET_LINK})
cuda_add_executable(texture texture.cu)

add_executable(buffer_object buffer_object.cpp)
target_link_libraries(buffer_object ${CUDA_TARGET_LINK} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${PNG_LIBRARIES})

cuda_add_executable(compat compat.cu)

cuda_add_executable(symbol symbol.cu)

cuda_add_executable(precision_sm_10 precision.cu)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--gpu-name;sm_13;--gpu-code;sm_13")
cuda_add_executable(precision_sm_13 precision.cu)

# run tests:

add_test(copy copy)
