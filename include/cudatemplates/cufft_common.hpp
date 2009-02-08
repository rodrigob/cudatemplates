/* 
  Cuda Templates.

  Copyright (C) 2008 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUFFT_COMMON_H
#define CUFFT_COMMON_H


#include <cufft.h>

#include <cudatemplates/error.hpp>


#define CUFFT_CHECK(call) { ::Cuda::FFT::result_t err = call; if(err != CUFFT_SUCCESS) CUDA_ERROR(::Cuda::FFT::getErrorString(err)); }


namespace Cuda {

/**
   This namespace wraps the CUFFT functionality.
*/
namespace FFT {

typedef cufftReal real;
typedef cufftComplex complex;
typedef cufftType_t type_t;
typedef cufftResult_t result_t;


/**
   Get CuFFT error string.
   Convert result code to error string according to the document
   "CUDA CUFFT Library", PG-00000-003_V2.0, April 2008.
   @param result numeric result code of CuFFT function call
   @return corresponding error string
*/
static const char *getErrorString(result_t result)
{
  switch(result) {
  case CUFFT_SUCCESS:         return "Any CUFFT operation is successful";
  case CUFFT_INVALID_PLAN:    return "CUFFT is passed an invalid plan handle";
  case CUFFT_ALLOC_FAILED:    return "CUFFT failed to allocate GPU memory.";
  case CUFFT_INVALID_TYPE:    return "The user requests an unsupported type.";
  case CUFFT_INVALID_VALUE:   return "The user specifies a bad memory pointer.";
  case CUFFT_INTERNAL_ERROR:  return "Used for all internal driver errors.";
  case CUFFT_EXEC_FAILED:     return "CUFFT failed to execute an FFT on the GPU.";
  case CUFFT_SETUP_FAILED:    return "The CUFFT library failed to initialize.";
    /*
      listed in the manual, but not defined in the header:
      case CUFFT_SHUTDOWN_FAILED: return "The CUFFT library failed to shut down.";
    */
  case CUFFT_INVALID_SIZE:    return "The user specifies an unsupported FFT size.";
  default:                    return "Unknown CUFFT error.";
  }
}

/**
   Generic FFT plan template.
   This template is empty, all behaviour is implemented in specializations of
   this template.
*/
template <class TypeIn, class TypeOut, unsigned Dim>
class Plan
{
};

}  // namespace FFT
}  // namespace Cuda


#endif
