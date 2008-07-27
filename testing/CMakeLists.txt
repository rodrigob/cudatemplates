set(Boost-Fixed_MINIMUM_VERSION 1.34)
find_package(Boost-Fixed REQUIRED)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories(../include ${Boost_INCLUDE_DIR})
add_definitions(-Wall)

add_executable(copy copy.cpp)
target_link_libraries(copy cuda cudart)

add_executable(fft fft.cpp)
target_link_libraries(fft cuda cudart cufft)

add_executable(itk itk.cpp)
target_link_libraries(itk cuda cudart ${ITK_LIBRARIES})
