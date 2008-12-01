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


#define CUBLAS_CHECK(call) { cublasStatus err = call; if(err != CUBLAS_STATUS_SUCCESS) throw Cuda::Error(__FILE__, __LINE__, __PRETTY_FUNCTION__, (int)err, 0); }
// #define CUBLAS_CHECK(call) call


namespace Cuda {

/**
   This namespace wraps the CUBLAS functionality.
*/
namespace BLAS {

typedef cuComplex complex;

/**
   CUBLAS initialization.
   Must be called once before using Cuda::BLAS.
*/
static inline void init()
{
  cublasInit();
}

template <class T>
class Vector
{
public:
  typedef T value_type;
  typedef unsigned size_type;

  inline int inc() const { return 1; }  // not yet used
  inline operator T*() { return m_devicePtr; }
  inline operator const T*() const { return m_devicePtr; }

  /*	
  inline T &operator()(size_type i)
  {
    return m_devicePtr[i];
  }
  */

  Vector():
    m_devicePtr(0), m_size(0)
  {
  }

  Vector(size_type size)
  {
    CUBLAS_CHECK(cublasAlloc(size, sizeof(T), (void **)&m_devicePtr));
    m_size = size;
    makeZero();
  }


  Vector(const Vector &v);


  ~Vector()
  {
    CUBLAS_CHECK(cublasFree(m_devicePtr));
    m_devicePtr = 0;
    m_size = 0;
  }


  void makeZero() {
    void* zeros = calloc(m_size, sizeof(T));
    CUBLAS_CHECK(cublasSetVector(m_size, sizeof(T), zeros, 1, m_devicePtr, 1));
    free(zeros);
  }


  void setValues(const T *values)
  {
    CUBLAS_CHECK(cublasSetVector(m_size, sizeof(T), values, 1, m_devicePtr, 1));
  }

  const T* getValues(void)
  {
    T* v = new T[m_size];
    CUBLAS_CHECK(cublasGetVector(m_size, sizeof(T), m_devicePtr, 1, v, 1));
    return v;
  }

  const size_type getSize() const
  {
    return m_size;
  }

private:
  size_type m_size;
  T *m_devicePtr;
};


template <class T>
class Matrix
{
public:
  typedef T value_type;
  typedef unsigned size_type;
	
  inline operator T*() { return m_devicePtr; }
  inline operator const T*() const { return m_devicePtr; }
	
  inline T &operator()(size_type i, size_type j) {
    return m_devicePtr[i * m_width + j];
  }
	
  inline T &operator()(size_type i, size_type j) const {
    return m_devicePtr[i * m_width + j];
  }
	
  Matrix():
    m_width(0), m_height(0), m_devicePtr(0)
  {
  }
	
  Matrix(size_type width, size_type height)
  {
    CUBLAS_CHECK(cublasAlloc(width * height, sizeof(T), (void **)&m_devicePtr));
    m_width = width;
    m_height = height;
  }
	
  ~Matrix()
  {
    CUBLAS_CHECK(cublasFree(m_devicePtr));
    m_width = 0;
    m_height = 0;
    m_devicePtr = 0;
  }

  /*	
  void setMatrix(const Matrix& host)
  {
    assert(m_width == host.getWidth() && m_height == host.getHeight());
    CUBLAS_CHECK(cublasSetMatrix(m_height, m_width, sizeof(T), host, m_width, m_devicePtr, m_width));
  }
  */

  void setValues(const T *values)
  {
    CUBLAS_CHECK(cublasSetMatrix(m_height, m_width, sizeof(T), values, m_width, m_devicePtr, m_width));
  }
	
  const size_type getWidth() const
  {
    return m_width;
  }
	
  const size_type getHeight() const
  {
    return m_height;
  }
	
private:
  size_type m_width;
  size_type m_height;
  T *m_devicePtr;
};

#include "cublas/blas1_float.hpp"
#include "cublas/blas1_complex.hpp"
#include "cublas/blas1_double.hpp"
#include "cublas/blas2_float.hpp"
#include "cublas/blas3_float.hpp"

template <class T>
	Vector<T>::Vector(const Vector<T> &v) {
	CUBLAS_CHECK(cublasAlloc(v.m_size, sizeof(T), (void **)&m_devicePtr));
	m_size = v.m_size;
	copy(v, *this);
}

}  // namespace BLAS
}  // namespace Cuda


#undef CUBLAS_CHECK


#endif
