# - Configure the CUDA templates library
# Once done this will define
#
#  CUDATEMPLATES_FOUND - system has CUDATEMPLATES
#  CUDATEMPLATES_DEFINITIONS - definitions for CUDATEMPLATES
#  CUDATEMPLATES_INCLUDE_DIR - the CUDATEMPLATES include directory
#

if(CUDATEMPLATES_INCLUDE_DIR)

  # in cache already
  set(CUDATEMPLATES_FOUND TRUE)

else()

  get_filename_component(CUDATEMPLATES_ROOT "${CMAKE_CURRENT_LIST_FILE}" PATH)

  if(NOT EXISTS ${CUDATEMPLATES_ROOT}/CMakeLists.txt)
    # we are *not* in the source tree, so find the actual root from here:
    # this is for Linux, Windows might be different:
    get_filename_component(CUDATEMPLATES_ROOT "${CUDATEMPLATES_ROOT}" PATH)
    get_filename_component(CUDATEMPLATES_ROOT "${CUDATEMPLATES_ROOT}" PATH)
  endif()

  set(CUDATEMPLATES_INCLUDE_DIR "${CUDATEMPLATES_ROOT}/include" CACHE PATH "The include directory of the CUDA templates")

  if(UNIX)
    set(CUDATEMPLATES_DEFINITIONS "-std=c++0x")
  endif()

  # always true since currently there is only template code:
  set(CUDATEMPLATES_FOUND TRUE)

  if(CUDATEMPLATES_FOUND)
    if(NOT CudaTemplates_FIND_QUIETLY)
      message(STATUS "Found CUDA templates: ${CUDATEMPLATES_INCLUDE_DIR}")
    endif()
  else()
    if(CudaTemplates_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find CUDA templates")
    endif()
  endif()

endif()
