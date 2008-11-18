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


#include <complex>

#include <cufft.h>

#include <cudatemplates/error.hpp>


#define CUFFT_CHECK(call) { ::Cuda::FFT::result_t err = call; if(err != CUFFT_SUCCESS) throw ::Cuda::Error(__FILE__, __LINE__, __PRETTY_FUNCTION__, (int)err, 0); }


namespace Cuda {

/**
   This namespace wraps the CUFFT functionality.
*/
namespace FFT {

typedef cufftReal real;
typedef std::complex<cufftReal> std_complex;
typedef cufftComplex complex;
typedef cufftType_t type_t;
typedef cufftResult_t result_t;

/**
   Generic FFT plan template.
   This template is empty, all behaviour is implemented specializations of this
   template.
*/
template <class TypeIn, class TypeOut, unsigned Dim>
class Plan
{
};

}  // namespace FFT
}  // namespace Cuda


#endif
