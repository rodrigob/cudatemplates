set(Boost-Fixed_MINIMUM_VERSION 1.34)
find_package(Boost-Fixed REQUIRED)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories(../include ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE} ${Boost_INCLUDE_DIR})
add_definitions(-Wall)

add_executable(copy copy.cpp)
target_link_libraries(copy ${CUDA_TARGET_LINK})

add_executable(fft fft.cpp)
target_link_libraries(fft ${CUDA_TARGET_LINK} cufft)

add_executable(itk itk.cpp)
target_link_libraries(itk ${CUDA_TARGET_LINK} ${ITK_LIBRARIES})

add_executable(array array.cpp)
target_link_libraries(array ${CUDA_TARGET_LINK})
