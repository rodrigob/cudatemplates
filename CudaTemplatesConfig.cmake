get_filename_component(CUDATEMPLATES_ROOT "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(NOT EXISTS ${CUDATEMPLATES_ROOT}/CMakeLists.txt)
  # We are *not* in the source tree, so find the actual root from here
  if(NOT WIN32)
    # this is for Linux, Windows might be different:
    get_filename_component(CUDATEMPLATES_ROOT "${CUDATEMPLATES_ROOT}" PATH)
    get_filename_component(CUDATEMPLATES_ROOT "${CUDATEMPLATES_ROOT}" PATH)
  endif(NOT WIN32)
endif(NOT EXISTS ${CUDATEMPLATES_ROOT}/CMakeLists.txt)

set(CUDATEMPLATES_INCLUDE_DIR "${CUDATEMPLATES_ROOT}/include")
set(CUDATEMPLATES_FOUND TRUE)
message(STATUS "Found CUDA Templates: ${CUDATEMPLATES_ROOT}")
