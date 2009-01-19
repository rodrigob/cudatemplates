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

#ifndef VECMULT_H
#define VECMULT_H


#include <complex>

#include <cudatemplates/devicememory.hpp>


namespace Cuda {

extern "C" {
  void vecmult_complex(int size, float *x1, const float *x2, const float *x3);
  void vecmult_complex_inplace(int size, float *x1, const float *x2);
  void vecmult_real(int size, float *x1, const float *x2, const float *x3);
  void vecmult_real_inplace(int size, float *x1, const float *x2);
}


inline void vecmult_complex(int size, std::complex<float> *x1, const std::complex<float> *x2, const std::complex<float> *x3)
{
  vecmult_complex(size, (float *)x1, (const float *)x2, (const float *)x3);
}

inline void vecmult_complex_inplace(int size, std::complex<float> *x1, const std::complex<float> *x2)
{
  vecmult_complex_inplace(size, (float *)x1, (const float *)x2);
}

template <class Type, unsigned Dim>
void vecmult_real_inplace(DeviceMemory<Type, Dim> &x1, const DeviceMemory<Type, Dim> &x2)
{
  if(x1.getSize() != x2.getSize())
    CUDA_ERROR("size mismatch");

  Cuda::vecmult_real_inplace(x1.getSize(), x1.getBuffer(), x2.getBuffer());
}

template <class Type, unsigned Dim>
void vecmult_real(DeviceMemory<Type, Dim> &x1, const DeviceMemory<Type, Dim> &x2, const DeviceMemory<Type, Dim> &x3)
{
  if((x1.getSize() != x2.getSize()) || (x1.getSize() != x3.getSize()))
    CUDA_ERROR("size mismatch");

  Cuda::vecmult_real(x1.getSize(), x1.getBuffer(), x2.getBuffer(), x3.getBuffer());
}

template <class Type, unsigned Dim>
void vecmult_complex_inplace(DeviceMemory<Type, Dim> &x1, const DeviceMemory<Type, Dim> &x2)
{
  if(x1.getSize() != x2.getSize())
    CUDA_ERROR("size mismatch");

  Cuda::vecmult_complex_inplace((int)x1.getSize(), x1.getBuffer(), x2.getBuffer());
}

template <class Type, unsigned Dim>
void vecmult_complex(DeviceMemory<Type, Dim> &x1, const DeviceMemory<Type, Dim> &x2, const DeviceMemory<Type, Dim> &x3)
{
  if((x1.getSize() != x2.getSize()) || (x1.getSize() != x3.getSize()))
    CUDA_ERROR("size mismatch");

  Cuda::vecmult_complex(x1.getSize(), x1.getBuffer(), x2.getBuffer(), x3.getBuffer());
}

}  // namespace Cuda


#endif
