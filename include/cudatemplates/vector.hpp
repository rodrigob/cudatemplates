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


#include <cudatemplates/error.hpp>
#include <cudatemplates/staticassert.hpp>


#define CUDA_VECTOR_OPS(vector_t, scalar_t)		                                                  \
  inline vector_t operator+ (const vector_t &x) const { vector_t res; add(res, *this, x); return res  ; } \
  inline vector_t operator+=(const vector_t &x)       {               add(     *this, x); return *this; } \
  inline vector_t operator- (const vector_t &x) const { vector_t res; sub(res, *this, x); return res  ; } \
  inline vector_t operator-=(const vector_t &x)       {               sub(     *this, x); return *this; } \
  inline vector_t operator* (      scalar_t  x) const { vector_t res; mul(res, *this, x); return res  ; } \
  inline vector_t operator*=(      scalar_t  x)       {               mul(     *this, x); return *this; } \
  inline vector_t operator/ (      scalar_t  x) const { vector_t res; div(res, *this, x); return res  ; } \
  inline vector_t operator/=(      scalar_t  x)       {               div(     *this, x); return *this; }


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

  /**
     Addition operator.
     @return sum of *this and given vector
     @param v vector to be added
  */
  /*
  inline VectorBase operator+(const VectorBase<Type, Dim> &v) const {
    VectorBase<Type, Dim> r;
    
    for(size_t i = Dim; i--;)
      r[i] = (*this)[i] + v[i];

    return r;
  }
  */

  /**
     Subtraction operator.
     Note that the size is unsigned in each dimension, i.e., you will get an
     overflow if you subtract a larger from a smaller quantity.
     @return difference of *this and given size
     @param v vector to be subtracted
  */
  /*
  inline VectorBase operator-(const VectorBase<Type, Dim> &v) const {
    VectorBase<Type, Dim> r;
    
    for(size_t i = Dim; i--;)
      r[i] = (*this)[i] - v[i];

    return r;
  }
  */

  /**
     Get total number of elements.
     This is the product of the sizes in each dimension.
     @return total number of elements
  */
  /*
  Type getVector() const {
    Type s = 1;

    for(int i = Dim; i--;)
      s *= data[i];

    return s;
  }
  */

  CUDA_VECTOR_OPS(VectorBase, Type);

protected:
  /**
     The size in each dimension.
  */
  Type data[Dim];

  /**
     Generic vector addition.
     Computes "v1+v2" and assigns result to v0.
     @param v0 result
     @param v1 first operand
     @param v2 second operand
  */
  inline static void add(VectorBase<Type, Dim> &v0,
			 const VectorBase<Type, Dim> &v1,
			 const VectorBase<Type, Dim> &v2)
  {
    for(size_t i = Dim; i--;)
      v0[i] = v1[i] + v2[i];
  }

  /**
     Generic vector addition (in-place).
     Computes "v1+v2" and assigns result to v1.
     @param v1 first operand and result
     @param v2 second operand
  */
  inline static void add(VectorBase<Type, Dim> &v1,
			 const VectorBase<Type, Dim> &v2)
  {
    for(size_t i = Dim; i--;)
      v1[i] += v2[i];
  }

  /**
     Generic vector subtraction.
     Computes "v1-v2" and assigns result to v0.
     @param v0 result
     @param v1 first operand
     @param v2 second operand
  */
  inline static void sub(VectorBase<Type, Dim> &v0,
			 const VectorBase<Type, Dim> &v1,
			 const VectorBase<Type, Dim> &v2)
  {
    for(size_t i = Dim; i--;)
      v0[i] = v1[i] - v2[i];
  }

  /**
     Generic vector subtraction (in-place).
     Computes "v1-v2" and assigns result to v1.
     @param v1 first operand and result
     @param v2 second operand
  */
  inline static void sub(VectorBase<Type, Dim> &v1,
			 const VectorBase<Type, Dim> &v2)
  {
    for(size_t i = Dim; i--;)
      v1[i] -= v2[i];
  }

  /**
     Generic vector-by-scalar multiplication.
     Computes "s2*v1" and assigns result to v0.
     @param v0 result (vector)
     @param v1 first operand (vector)
     @param s2 second operand (scalar)
  */
  inline static void mul(VectorBase<Type, Dim> &v0,
			 const VectorBase<Type, Dim> &v1,
			 Type s2)
  {
    for(size_t i = Dim; i--;)
      v0[i] = v1[i] * s2;
  }

  /**
     Generic vector-by-scalar multiplication (in-place).
     Computes "s2*v1" and assigns result to v1.
     @param v1 first operand and result (vector)
     @param s2 second operand (scalar)
  */
  inline static void mul(VectorBase<Type, Dim> &v1,
			 Type s2)
  {
    for(size_t i = Dim; i--;)
      v1[i] *= s2;
  }

  /**
     Generic vector-by-scalar division.
     Computes "(1/s2)*v1" and assigns result to v0.
     @param v0 result (vector)
     @param v1 first operand (vector)
     @param s2 second operand (scalar)
  */
  inline static void div(VectorBase<Type, Dim> &v0,
			 const VectorBase<Type, Dim> &v1,
			 Type s2)
  {
    for(size_t i = Dim; i--;)
      v0[i] = v1[i] / s2;
  }

  /**
     Generic vector-by-scalar division (in-place).
     Computes "(1/s2)*v1" and assigns result to v1.
     @param v1 first operand and result (vector)
     @param s2 second operand (scalar)
  */
  inline static void div(VectorBase<Type, Dim> &v1,
			 Type s2)
  {
    for(size_t i = Dim; i--;)
      v1[i] /= s2;
  }
};

/**
   Generic vector template.
*/
template <class Type, unsigned Dim>
class Vector: public VectorBase<Type, Dim>
{
  Vector() {}
};

/**
   Specialization of vector template for 1D case.
*/
template <class Type>
class Vector<Type, 1>: public VectorBase<Type, 1>
{
public:
  /**
     Default constructor.
  */
  Vector() {}

  /**
     Constructor.
     @param x value in x-direction
  */
  Vector(Type x) { this->data[0] = x; }

  CUDA_VECTOR_OPS(Vector, Type);
};

/**
   Specialization of vector template for 2D case.
*/
template <class Type>
class Vector<Type, 2>: public VectorBase<Type, 2>
{
public:
  /**
     Default constructor.
  */
  Vector() {}

  /**
     Constructor.
     @param x value in x-direction
     @param y value in y-direction
  */
  Vector(Type x, Type y) { this->data[0] = x; this->data[1] = y; }

  CUDA_VECTOR_OPS(Vector, Type);
};

/**
   Specialization of vector template for 3D case.
*/
template <class Type>
class Vector<Type, 3>: public VectorBase<Type, 3>
{
public:
  /**
     Default constructor.
  */
  Vector() {}

  /**
     Constructor.
     @param x value in x-direction
     @param y value in y-direction
     @param z value in z-direction
  */
  Vector(Type x, Type y, Type z) { this->data[0] = x; this->data[1] = y; this->data[2] = z; }

  CUDA_VECTOR_OPS(Vector, Type);
};

/**
   Equality operator.
   @param v1 first size
   @param v2 second size
   @return true if first and second size are equal, otherwise false
*/
template <class Type, unsigned Dim>
bool operator==(const VectorBase<Type, Dim> &v1, const VectorBase<Type, Dim> &v2)
{
  for(int i = Dim; i--;)
    if(v1[i] != v2[i])
      return false;

  return true;
}

/**
   Inequality operator.
   @param v1 first size
   @param v2 second size
   @return true if first and second size are not equal, otherwise false
*/
template <class Type, unsigned Dim>
bool operator!=(const VectorBase<Type, Dim> &v1, const VectorBase<Type, Dim> &v2)
{
  for(size_t i = Dim; i--;)
    if(v1[i] != v2[i])
      return true;

  return false;
}

/**
   Division operator.
   Note that if a division with a factor 0 is executed the behaviour is not defined.
   @param lhs dividend
   @param rhs divisor
   @return qotient of division of \a lhs (dividend) and given factor \a rhs (divisor)
*/
/*
template <class Type, unsigned Dim>
Cuda::Vector<Type, Dim> operator/(const Cuda::Vector<Type, Dim> &lhs, const Type &rhs)
{
  if (rhs == 0)
    CUDA_ERROR("Division by zero");
  Cuda::Vector<Type, Dim> out = lhs;
  for(size_t i = 0; i < Dim; ++i)
    out[i] /= rhs;
  
  return out;
}
*/

// /**
//    Subtraction Assignment operator.
//    Note that the size is unsigned in each dimension, i.e., you will get an
//    overflow if you subtract a larger from a smaller quantity.
//    @param other subtrahend size to be subtracted
//    @return difference of \a this and \a other
// */
// template <unsigned Dim>
// Cuda::Vector<Dim>& operator-=(const Cuda::Vector<Dim> &other) {
//   Cuda::Vector<Dim> out;
//   for(size_t i = Dim; i--;)
//     out[i]= (*this)[i] - other[i];
//   return out;
// }

/**
   Subtraction operator.
   Note that the size is unsigned in each dimension, i.e., you will get an
   overflow if you subtract a larger from a smaller quantity.
   @param lhs minuend size of subtraction
   @param rhs subtrahend size to be subtracted
   @return difference of \a lhs and \a rhs
*/
 /*
template <class Type, unsigned Dim>
Cuda::Vector<Type, Dim> operator-(const Cuda::Vector<Type, Dim> &lhs, const Cuda::Vector<Type, Dim> &rhs) {
  Cuda::Vector<Type, Dim> out = lhs;
  for(size_t i = Dim; i--;)
    out[i]-= rhs[i];
  return out;
}
 */


}  // namespace Cuda


#endif
