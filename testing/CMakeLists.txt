find_package(Boost 1.34 REQUIRED)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories(../include ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE} ${Boost_INCLUDE_DIR})
cuda_include_directories(../include ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE})
add_definitions(-Wall)

#set(CUDA_BUILD_TYPE Device)
set(CUDA_BUILD_TYPE Emulation)

add_executable(copy copy.cpp)
target_link_libraries(copy ${CUDA_TARGET_LINK})

add_executable(fft fft.cpp)
target_link_libraries(fft ${CUDA_TARGET_LINK} cufft)

add_executable(itk itk.cpp)
target_link_libraries(itk ${CUDA_TARGET_LINK} ${ITK_LIBRARIES})

add_executable(gil gil.cpp)
target_link_libraries(gil ${CUDA_TARGET_LINK} png)

add_executable(array array.cpp)
target_link_libraries(array ${CUDA_TARGET_LINK})

cuda_add_executable(kernel kernel.cu)

cuda_add_executable(texture texture.cu)
#add_executable(texture texture.cpp)
#target_link_libraries(texture ${CUDA_TARGET_LINK})
