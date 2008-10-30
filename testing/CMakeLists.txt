cmake_minimum_required(VERSION 2.6)
# set icg cmake module path
SET(CMAKE_MODULE_PATH $ENV{ICG_CMAKE_MODULE_PATH})

# Boost:
find_package(Boost 1.34 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Cuda:
set(CUDA_BUILD_CUBIN OFF)
set(CUDA_BUILD_TYPE Device)
#set(CUDA_BUILD_TYPE Emulation)
find_package(Cuda REQUIRED)
include_directories(${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE})

# cudatemplates
FIND_PACKAGE(CudaTemplates REQUIRED)
include_directories(${CUDATEMPLATES_INCLUDE_DIR})
cuda_include_directories(${CUDATEMPLATES_INCLUDE_DIR})

# ITK:
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# OpenCV:
find_package( OpenCV REQUIRED )
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

add_executable(fft fft.cpp)
target_link_libraries(fft ${CUDA_TARGET_LINK} ${CUFFT_TARGET_LINK})

add_executable(blas blas.cpp)
target_link_libraries(blas ${CUDA_TARGET_LINK} ${CUBLAS_TARGET_LINK})

add_executable(itk itk.cpp)
target_link_libraries(itk ${CUDA_TARGET_LINK} ${ITK_LIBRARIES})

add_executable(gil gil.cpp)
target_link_libraries(gil ${CUDA_TARGET_LINK} ${PNG_LIBRARIES})

add_executable(ipl ipl.cpp)
target_link_libraries(ipl ${CUDA_TARGET_LINK} ${OPENCV_LIBRARIES})

add_executable(array array.cpp)
target_link_libraries(array ${CUDA_TARGET_LINK})

cuda_add_executable(kernel kernel.cu)
target_link_libraries(kernel ${CUDA_TARGET_LINK})

add_executable(texture texture.cu)
target_link_libraries(texture ${CUDA_TARGET_LINK})
cuda_add_executable(texture texture.cu)

 add_executable(buffer_object buffer_object.cpp)
 target_link_libraries(buffer_object ${CUDA_TARGET_LINK} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${PNG_LIBRARIES})
