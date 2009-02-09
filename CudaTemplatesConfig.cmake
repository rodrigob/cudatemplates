# - Configure the CUDA templates library
# Once done this will define
#
#  CUDATEMPLATES_FOUND - system has CUDATEMPLATES
#  CUDATEMPLATES_DEFINITIONS - definitions for CUDATEMPLATES
#  CUDATEMPLATES_INCLUDE_DIR - the CUDATEMPLATES include directory
#  CUDATEMPLATES_LIBRARIES - Link these to use CUDATEMPLATES
#  CUDATEMPLATES_LIBRARY_DIRS - link directories, useful for rpath
#

if(CUDATEMPLATES_INCLUDE_DIR AND CUDATEMPLATES_LIBRARY_DIRS)

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
  set(CUDATEMPLATES_LIBRARIES "cudatemplates" CACHE STRING "CUDA templates library name")

  # search for the library (if we are in the source tree, it might not exist):
  find_library(CUDATEMPLATES_LIBRARY_PATH "${CUDATEMPLATES_LIBRARIES}" PATHS
    "${CUDATEMPLATES_ROOT}/src"
    "${CUDATEMPLATES_ROOT}/lib"
    "${CUDATEMPLATES_ROOT}/lib/Release"
    "${CUDATEMPLATES_ROOT}/lib${LIBSUFFIX}"
    "${CUDATEMPLATES_ROOT}/lib64"
    "${CUDATEMPLATES_ROOT}/lib/win32"
    # make sure not to mix locally installed headers with globally installed binaries:
    NO_DEFAULT_PATH
  )

  if(CUDATEMPLATES_LIBRARY_PATH)
    # store library directories in cache:
    get_filename_component(CUDATEMPLATES_LIBRARY_DIRS "${CUDATEMPLATES_LIBRARY_PATH}" PATH)
    set(CUDATEMPLATES_LIBRARY_DIRS "${CUDATEMPLATES_LIBRARY_DIRS}" CACHE STRING "CUDA templates library directories")
    set(CUDATEMPLATES_FOUND TRUE)
  endif()

  if(CUDATEMPLATES_FOUND)
    if(NOT CudaTemplates_FIND_QUIETLY)
      message(STATUS "Found CUDA templates: ${CUDATEMPLATES_LIBRARY_DIRS} ${CUDATEMPLATES_INCLUDE_DIR}")
    endif()
  else()
    if(CudaTemplates_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find CUDA templates")
    endif()
  endif()

endif()
