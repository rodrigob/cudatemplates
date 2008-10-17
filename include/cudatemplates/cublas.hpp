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

#ifndef CUBLASPP_H
#define CUBLASPP_H


#include <cassert>
#include <complex>

#include <cublas.h>

#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/error.hpp>


//!!! #define CUBLAS_CHECK(call) { result_t_t err = call; if(err != CUBLAS_SUCCESS) throw Cuda::Error(__FILE__, __LINE__, __PRETTY_FUNCTION__, (int)err, 0); }
#define CUBLAS_CHECK(call) call


namespace Cuda {
namespace BLAS {

typedef cuComplex complex;

static inline void init()
{
  cublasInit();
}

template <class T, int N>
class Vector
{
public:
  Vector()
  {
    CUBLAS_CHECK(cublasAlloc(N, sizeof(T), (void **)&devicePtr));
  }

  ~Vector()
  {
    CUBLAS_CHECK(cublasFree(devicePtr));
  }

  inline int inc() const { return 1; }  // not yet used
  inline operator T *() { return devicePtr; }
  inline operator const T *() const { return devicePtr; }

private:
  T *devicePtr;
};


template <class T, int M, int N>
class Matrix
{
public:
  Matrix()
  {
    CUBLAS_CHECK(cublasAlloc(N * M, sizeof(T), (void **)&devicePtr));
  }

  ~Matrix()
  {
    CUBLAS_CHECK(cublasFree(devicePtr));
  }

  inline operator T *() { return devicePtr; }
  inline operator const T *() const { return devicePtr; }

private:
  T *devicePtr;
};


#include "cublas/blas1_float.hpp"
#include "cublas/blas1_complex.hpp"
#include "cublas/blas1_double.hpp"
#include "cublas/blas2_float.hpp"
#include "cublas/blas3_float.hpp"


}  // namespace BLAS
}  // namespace Cuda


#undef CUBLAS_CHECK


#endif
