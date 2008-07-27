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


#include <cudatemplates/staticassert.hpp>


namespace Cuda {

/**
   Base class for multi-dimensional size type.
   Think of it as a multi-dimensional variant of size_t.
*/
template <unsigned Dim>
class SizeBase
{
  CUDA_STATIC_ASSERT(Dim >= 1);

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
  */
  inline size_t operator[](size_t i) const { return size[i]; }

  /**
     Array index operator.
  */
  inline size_t &operator[](size_t i) { return size[i]; }

  /**
     Addition operator.
  */
  inline SizeBase operator+(const SizeBase<Dim> &s) const {
    SizeBase<Dim> r;
    
    for(int i = Dim; i--;)
      r[i] = (*this)[i] + s[i];

    return r;
  }

  /**
     Subtraction operator.
  */
  inline SizeBase operator-(const SizeBase<Dim> &s) const {
    SizeBase<Dim> r;
    
    for(int i = Dim; i--;)
      r[i] = (*this)[i] - s[i];

    return r;
  }

  /**
     Get total number of elements.
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
*/
template <unsigned Dim>
bool operator!=(const SizeBase<Dim> &s1, const SizeBase<Dim> &s2)
{
  for(int i = Dim; i--;)
    if(s1[i] != s2[i])
      return true;

  return false;
}

}


#endif
