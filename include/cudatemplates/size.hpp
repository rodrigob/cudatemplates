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

#ifndef CUDA_SIZE_H
#define CUDA_SIZE_H


// #define CUDA_USE_OLD_SIZE


#ifdef WIN32
#pragma warning(disable: 4127)  // "conditional expression is constant": yes, this is called "template metaprogramming"
#pragma warning(disable: 4710)  // "function not inlined": what exactly is the problem here?
#endif


#include <cudatemplates/staticassert.hpp>

#ifndef CUDA_USE_OLD_SIZE
#include <cudatemplates/vector.hpp>
#endif


namespace Cuda {

#ifdef CUDA_USE_OLD_SIZE

/**
   Base class for multi-dimensional size type.
   Think of it as a multi-dimensional variant of size_t.
*/
template <unsigned Dim>
class SizeBase
{
  CUDA_STATIC_ASSERT(Dim > 0);

public:
  /**
     Default constructor.
     Set size to zero in all dimensions.
  */
  SizeBase()
  {
    for(int i = Dim; i--;)
      size[i] = 0;
  }

  /**
     Array index operator.
     @return size in given (i-th) dimension
     @param i index
  */
  inline size_t operator[](size_t i) const { return size[i]; }

  /**
     Array index operator.
     @return size in given (i-th) dimension
     @param i index
  */
  inline size_t &operator[](size_t i) { return size[i]; }

  /**
     Addition operator.
     @return sum of *this and given size
     @param s size to be added
  */
  inline SizeBase operator+(const SizeBase<Dim> &s) const {
    SizeBase<Dim> r;
    
    for(size_t i = Dim; i--;)
      r[i] = (*this)[i] + s[i];

    return r;
  }

  /**
     Subtraction operator.
     Note that the size is unsigned in each dimension, i.e., you will get an
     overflow if you subtract a larger from a smaller quantity.
     @return difference of *this and given size
     @param s size to be subtracted
  */
  inline SizeBase operator-(const SizeBase<Dim> &s) const {
    SizeBase<Dim> r;
    
    for(size_t i = Dim; i--;)
      r[i] = (*this)[i] - s[i];

    return r;
  }

  /**
     Get total number of elements.
     This is the product of the sizes in each dimension.
     @return total number of elements
  */
  size_t getSize() const {
    size_t s = 1;

    for(int i = Dim; i--;)
      s *= size[i];

    return s;
  }

protected:
  /**
     The size in each dimension.
  */
  size_t size[Dim];
};

/**
   Generic size template.
*/
template <unsigned Dim>
class Size: public SizeBase<Dim>
{
  Size() {}
};

/**
   Specialization of size template for 1D case.
*/
template <>
class Size<1>: public SizeBase<1>
{
public:
  /**
     Default constructor.
  */
  Size() {}

  /**
     Constructor.
     @param s0 size in x-direction
  */
  Size(size_t s0) { size[0] = s0; }
};

/**
   Specialization of size template for 2D case.
*/
template <>
class Size<2>: public SizeBase<2>
{
public:
  /**
     Default constructor.
  */
  Size() {}

  /**
     Constructor.
     @param s0 size in x-direction
     @param s1 size in y-direction
  */
  Size(size_t s0, size_t s1) { size[0] = s0; size[1] = s1; }
};

/**
   Specialization of size template for 3D case.
*/
template <>
class Size<3>: public SizeBase<3>
{
public:
  /**
     Default constructor.
  */
  Size() {}

  /**
     Constructor.
     @param s0 size in x-direction
     @param s1 size in y-direction
     @param s2 size in z-direction
  */
  Size(size_t s0, size_t s1, size_t s2) { size[0] = s0; size[1] = s1; size[2] = s2; }
};

/**
   Equality operator.
   @param s1 first size
   @param s2 second size
   @return true if first and second size are equal, otherwise false
*/
template <unsigned Dim>
bool operator==(const SizeBase<Dim> &s1, const SizeBase<Dim> &s2)
{
  for(int i = Dim; i--;)
    if(s1[i] != s2[i])
      return false;

  return true;
}

/**
   Inequality operator.
   @param s1 first size
   @param s2 second size
   @return true if first and second size are not equal, otherwise false
*/
template <unsigned Dim>
bool operator!=(const SizeBase<Dim> &s1, const SizeBase<Dim> &s2)
{
  for(size_t i = Dim; i--;)
    if(s1[i] != s2[i])
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
template <unsigned Dim>
Cuda::Size<Dim> operator/(const Cuda::Size<Dim> &lhs, const float &rhs) {
  if (rhs == 0)
    fprintf(stderr, "Division by zero!!\n");
  Cuda::Size<Dim> out = lhs;
  for(size_t i = 0; i < Dim; ++i)
    out[i] /= rhs;
  
  return out;
}

/**
   Subtraction operator.
   Note that the size is unsigned in each dimension, i.e., you will get an
   overflow if you subtract a larger from a smaller quantity.
   @param lhs minuend size of subtraction
   @param rhs subtrahend size to be subtracted
   @return difference of \a lhs and \a rhs
*/
template <unsigned Dim>
Cuda::Size<Dim> operator-(const Cuda::Size<Dim> &lhs, const Cuda::Size<Dim> &rhs) {
  Cuda::Size<Dim> out = lhs;
  for(size_t i = Dim; i--;)
    out[i]-= rhs[i];
  return out;
}

#else  // CUDA_USE_OLD_SIZE

#ifdef WIN32
#define ssize_t int
#endif

/**
   Base class for multi-dimensional size type.
   Think of it as a multi-dimensional variant of size_t.
*/
template <unsigned Dim>
class SizeBase: public VectorBase<size_t, Dim>
{
public:
  /**
     Get total number of elements.
     This is the product of the sizes in each dimension.
     @return total number of elements
  */
  size_t getSize() const {
    size_t s = 1;

    for(int i = Dim; i--;)
      s *= this->data[i];

    return s;
  }
};

/**
   Generic spacing template
*/
template<unsigned Dim>
class Spacing : public VectorBase<float, Dim>
{
};

/**
   Generic size template.
*/
template <unsigned Dim>
class Size: public SizeBase<Dim>
{
};

/**
   Base class for multi-dimensional ssize type.
   Think of it as a multi-dimensional variant of ssize_t.
*/
template <unsigned Dim>
class SSizeBase: public VectorBase<ssize_t, Dim>
{
};

/**
   Generic size template.
*/
template <unsigned Dim>
class SSize: public SSizeBase<Dim>
{
  SSize() {}
};


#endif  // CUDA_USE_OLD_SIZE

}  // namespace Cuda


#ifndef CUDA_USE_OLD_SIZE
#include "auto/specdim_vector_size.hpp"
#include "auto/specdim_vector_ssize.hpp"
#endif


#endif
