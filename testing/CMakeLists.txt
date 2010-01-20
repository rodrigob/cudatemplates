# Cuda:
set(CUDA_BUILD_CUBIN OFF)
set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
#set(CUDA_BUILD_EMULATION ON CACHE BOOL "enable emulation mode")
find_package(CUDA 2.2 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})

# if you get error messages in nvcc-generated files,
# enable the following line for debugging:
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--keep")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fpermissive")

# CUDA templates:
include_directories(${CMAKE_SOURCE_DIR}/include)
cuda_include_directories(${CMAKE_SOURCE_DIR}/include)

# Glew:
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})
cuda_include_directories(${GLEW_INCLUDE_DIR})

# Boost:
find_package(Boost)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIR})
  cuda_include_directories(${Boost_INCLUDE_DIR})
endif(Boost_FOUND)

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

# ImageMagick:
#find_package(ImageMagick COMPONENTS Magick++)
#include_directories(${ImageMagick_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)
cuda_include_directories(${CMAKE_SOURCE_DIR}/include)

#add_definitions(-Wall)

# There seems to be a problem with gcc < 4.3 and c++0x. Therefore we define the -std only if gcc >= 4.3
# TODO: workarounf for gcc < 4.3
if(UNIX)
  exec_program(gcc ARGS --version OUTPUT_VARIABLE CMAKE_C_COMPILER_VERSION)
  if(CMAKE_C_COMPILER_VERSION MATCHES ".*4\\.[3-9].*")
    message(STATUS "GCC >= 4.3")
    add_definitions(-std=c++0x)
  else()
    message(STATUS ${CMAKE_C_COMPILER_VERSION})
  endif()
endif(UNIX)


if(WINDOWS)
  # TODO: enable "decltype" keyword for Windows (unless enabled by default)
endif(WINDOWS)


# test programs:

add_executable(array array.cpp)
target_link_libraries(array ${CUDA_LIBRARIES})

cuda_add_executable(array_by_value array_by_value.cu)

add_executable(blas blas.cpp)
target_link_libraries(blas ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(border border.cpp)
target_link_libraries(border ${CUDA_LIBRARIES})

cuda_add_executable(buffer_object buffer_object.cpp buffer_object_init.cu)
target_link_libraries(buffer_object ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${PNG_LIBRARIES})

cuda_add_executable(color_speed_test color_speed_test.cu)

cuda_add_executable(copy copy.cpp copy_instantiate.cu)

cuda_add_executable(convert convert.cu)

add_executable(demo demo.cpp)
target_link_libraries(demo ${CUDA_LIBRARIES})

add_executable(fft fft.cpp)
target_link_libraries(fft ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})

cuda_add_executable(fillrate fillrate.cu)

if(Boost_FOUND)
  add_executable(gil gil.cpp)
  target_link_libraries(gil ${CUDA_LIBRARIES} ${PNG_LIBRARIES})
endif(Boost_FOUND)

if(OpenCV_FOUND)
  add_executable(ipl ipl.cpp)
  target_link_libraries(ipl ${CUDA_LIBRARIES} ${OPENCV_LIBRARIES})
endif(OpenCV_FOUND)

if(ITK_FOUND)
  add_executable(itk itk.cpp)
  target_link_libraries(itk ${CUDA_LIBRARIES} ${ITK_LIBRARIES})
endif(ITK_FOUND)

cuda_add_executable(kernel kernel.cu)
target_link_libraries(kernel ${CUDA_LIBRARIES})

cuda_add_executable(mapped mapped.cu)

add_executable(memset memset.cpp)
target_link_libraries(memset ${CUDA_LIBRARIES})

cuda_add_executable(ogl_texture ogl_texture.cu)
target_link_libraries(ogl_texture ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})

cuda_add_executable(pack pack.cu)
add_dependencies(pack create_pack)

cuda_add_executable(render render.cu)
target_link_libraries(render ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(streams streams.cu)

# cuda_add_executable(surface surface.cu)

cuda_add_executable(symbol symbol.cu)

cuda_add_executable(symbol_plain symbol_plain.cu)

cuda_add_executable(texture texture.cu)

cuda_add_executable(texture_precision texture_precision.cu)

cuda_add_executable(throughput throughput.cu)

# cuda_add_executable(vector vector.cpp)

if(WIN32)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}
/NODEFAULTLIB:LIBCMT.lib ")
endif(WIN32)
#cuda_add_executable(crazyFindOGLTextureAdress crazyFindOGLTextureAdress.cu crazyFindOGLTextureAdress.cpp)
#target_link_libraries(crazyFindOGLTextureAdress ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${PNG_LIBRARIES})


#cuda_add_executable(precision_sm_10 precision.cu)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--gpu-name;sm_13;--gpu-code;sm_13")
#cuda_add_executable(precision_sm_13 precision.cu)

# not working with CUDA-3.0-beta1:
if(CUDA_VERSION VERSION_LESS 3.0)
  cuda_add_executable(compat compat.cu)
endif()

# run tests:

add_test(copy copy)
