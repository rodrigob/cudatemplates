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

#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H


#ifdef WIN32
#pragma warning(disable: 4127)  // "conditional expression is constant": yes, this is called "template metaprogramming"
#pragma warning(disable: 4710)  // "function not inlined": what exactly is the problem here?
#endif


#if defined(__GXX_EXPERIMENTAL_CXX0X__) || defined(__HOWEVER_THIS_IS_CALLED_FOR_WINDOWS__)
#define CUDA_HAS_DECLTYPE 1
#else
#define CUDA_HAS_DECLTYPE 0
#endif


/*
  In device emulation mode, nvcc can't access vector elements in a template
  kernel. We need some tricky (and ugly) workaround to achieve this.
*/
#if __DEVICE_EMULATION__
#define CUDA_EMULATION_VECTOR_ACCESS_WORKAROUND 1
#define CUDA_KERNEL_SIZE(n) Cuda::VectorBase<size_t, n>
#else
#define CUDA_EMULATION_VECTOR_ACCESS_WORKAROUND 0
#define CUDA_KERNEL_SIZE(n) Cuda::Size<n>
#endif


#include <cudatemplates/error.hpp>
#include <cudatemplates/staticassert.hpp>


namespace Cuda {

/**
   Base class for vector type.
*/
template <class Type, unsigned Dim>
class VectorBase
{
  CUDA_STATIC_ASSERT(Dim > 0);

public:
  /**
     Default constructor.
     Set size to zero in all dimensions.
  */
  VectorBase()
  {
    for(int i = Dim; i--;)
      data[i] = 0;
  }

  /**
     Array index operator.
     @return size in given (i-th) dimension
     @param i index
  */
  inline Type operator[](size_t i) const { return data[i]; }

  /**
     Array index operator.
     @return size in given (i-th) dimension
     @param i index
  */
  inline Type &operator[](size_t i) { return data[i]; }

#if CUDA_EMULATION_VECTOR_ACCESS_WORKAROUND
public:
#else
protected:
#endif
  /**
     The size in each dimension.
  */
  Type data[Dim];
};

/**
   Generic vector template.
*/
template <class Type, unsigned Dim>
class Vector: public VectorBase<Type, Dim>
{
public:
  Vector() {}
};

/**
   Equality operator.
   @param v1 first vector
   @param v2 second vector
   @return true if first and second vector are equal, otherwise false
*/
template <class Type1, class Type2, unsigned Dim>
bool
operator==(const VectorBase<Type1, Dim> &v1, const VectorBase<Type2, Dim> &v2)
{
  for(unsigned i = Dim; i--;)
    if(v1[i] != v2[i])
      return false;

  return true;
}

/**
   Inequality operator.
   @param v1 first vector
   @param v2 second vector
   @return true if first and second vector are not equal, otherwise false
*/
template <class Type1, class Type2, unsigned Dim>
bool
operator!=(const VectorBase<Type1, Dim> &v1, const VectorBase<Type2, Dim> &v2)
{
  for(unsigned i = Dim; i--;)
    if(v1[i] != v2[i])
      return true;

  return false;
}

#if CUDA_HAS_DECLTYPE

#define EXPR_TYPE(op1, op, op2) decltype((op1)0 op (op2)0)

template <class Type1, class Type2, unsigned Dim>
VectorBase<EXPR_TYPE(Type1, +, Type2), Dim>
operator+(const VectorBase<Type1, Dim> &v1, const VectorBase<Type2, Dim> &v2)
{
  VectorBase<EXPR_TYPE(Type1, +, Type2), Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = v1[i] + v2[i];

  return v;
}

template <class Type1, class Type2, unsigned Dim>
VectorBase<EXPR_TYPE(Type1, -, Type2), Dim>
operator-(const VectorBase<Type1, Dim> &v1, const VectorBase<Type2, Dim> &v2)
{
  VectorBase<EXPR_TYPE(Type1, -, Type2), Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = v1[i] - v2[i];

  return v;
}

template <class Type1, class Type2, unsigned Dim>
VectorBase<EXPR_TYPE(Type1, *, Type2), Dim>
operator*(Type1 s1, const VectorBase<Type2, Dim> &v2)
{
  VectorBase<EXPR_TYPE(Type1, *, Type2), Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = s1 * v2[i];

  return v;
}

template <class Type1, class Type2, unsigned Dim>
VectorBase<EXPR_TYPE(Type1, *, Type2), Dim>
operator*(const VectorBase<Type1, Dim> &v1, Type2 s2)
{
  VectorBase<EXPR_TYPE(Type1, *, Type2), Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = v1[i] * s2;

  return v;
}

template <class Type1, class Type2, unsigned Dim>
VectorBase<EXPR_TYPE(Type1, /, Type2), Dim>
operator/(const VectorBase<Type1, Dim> &v1, Type2 s2)
{
  VectorBase<EXPR_TYPE(Type1, /, Type2), Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = v1[i] / s2;

  return v;
}

#else  // CUDA_HAS_DECLTYPE

template <class Type, unsigned Dim>
VectorBase<Type, Dim>
operator+(const VectorBase<Type, Dim> &v1, const VectorBase<Type, Dim> &v2)
{
  VectorBase<Type, Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = v1[i] + v2[i];

  return v;
}

template <class Type, unsigned Dim>
VectorBase<Type, Dim>
operator-(const VectorBase<Type, Dim> &v1, const VectorBase<Type, Dim> &v2)
{
  VectorBase<Type, Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = v1[i] - v2[i];

  return v;
}

template <class Type, unsigned Dim>
VectorBase<Type, Dim>
operator*(Type s1, const VectorBase<Type, Dim> &v2)
{
  VectorBase<Type, Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = s1 * v2[i];

  return v;
}

template <class Type, unsigned Dim>
VectorBase<Type, Dim>
operator*(const VectorBase<Type, Dim> &v1, Type s2)
{
  VectorBase<Type, Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = v1[i] * s2;

  return v;
}

template <class Type, unsigned Dim>
VectorBase<Type, Dim>
operator/(const VectorBase<Type, Dim> &v1, Type s2)
{
  VectorBase<Type, Dim> v;

  for(unsigned i = Dim; i--;)
    v[i] = v1[i] / s2;

  return v;
}

#endif  // CUDA_HAS_DECLTYPE

}  // namespace Cuda


#include "auto/specdim_vector_vector.hpp"


#endif
