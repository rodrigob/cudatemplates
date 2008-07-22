set(Boost-Fixed_MINIMUM_VERSION 1.34)
find_package(Boost-Fixed REQUIRED)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories(../include ${Boost_INCLUDE_DIR})
add_definitions(-Wall)

add_executable(cpptest main.cpp)
target_link_libraries(cpptest cuda cudart)

add_executable(itktest itktest.cpp)
target_link_libraries(itktest cuda cudart ${ITK_LIBRARIES})
