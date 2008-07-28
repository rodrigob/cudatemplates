# - Try to find the CUDA Templates library
# Once done this will define
#
#  CUDATEMPLATES_FOUND - system has CUDATEMPLATES
#  CUDATEMPLATES_INCLUDE_DIR - the CUDATEMPLATES include directory
#

if(CUDATEMPLATES_INCLUDE_DIR)
  # in cache already
  set(CUDATEMPLATES_FOUND TRUE)
else(CUDATEMPLATES_INCLUDE_DIR)
  if(WIN32)
    # check whether the CUDATEMPLATESDIR environment variable is set and points to a 
    # valid windows CUDATEMPLATES installation
    # ...
  else(WIN32)
    find_path(CUDATEMPLATES_INCLUDE_DIR cudatemplates/copy.hpp /usr/include /usr/local/include)
    if(CUDATEMPLATES_INCLUDE_DIR)
      set(CUDATEMPLATES_FOUND TRUE)
    endif(CUDATEMPLATES_INCLUDE_DIR)
  endif(WIN32)
  if(CUDATEMPLATES_FOUND)
    if(NOT CudaTemplates_FIND_QUIETLY)
      message(STATUS "Found CUDATEMPLATES: ${CUDATEMPLATES_LIBRARIES}")
    endif(NOT CudaTemplates_FIND_QUIETLY)
  else(CUDATEMPLATES_FOUND)
    if(CudaTemplates_FIND_REQUIRED)
      message(FATAL_ERROR "Could NOT find CUDA Templates")
    endif(CudaTemplates_FIND_REQUIRED)
  endif(CUDATEMPLATES_FOUND)
endif(CUDATEMPLATES_INCLUDE_DIR)
