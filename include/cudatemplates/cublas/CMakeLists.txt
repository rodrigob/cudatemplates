set(warning "/*\n  Note to cudatemplates developers:\n  don't modify this file directly, modify the corresponding \"*.hpp.in\" file instead and re-run cmake!\n*/")

set(type "float")
set(t "s")
set(T "S")
configure_file(blas1.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/blas1_${type}.hpp)
configure_file(blas2.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/blas2_${type}.hpp)
configure_file(blas3.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/blas3_${type}.hpp)

set(type "complex")
set(t "c")
set(T "C")
configure_file(blas1.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/blas1_${type}.hpp)
configure_file(blas2.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/blas2_${type}.hpp)

set(type "double")
set(t "d")
set(T "D")
configure_file(blas1.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/blas1_${type}.hpp)
