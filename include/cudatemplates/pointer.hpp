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

#ifndef CUDA_POINTER_H
#define CUDA_POINTER_H


#include <cudatemplates/iterator.hpp>
#include <cudatemplates/layout.hpp>


namespace Cuda {

/**
   Class to represent memory that is accessible via a pointer.
   This is used as a virtual base class for all types of memory which can be
   accessed via a typed pointer. Host and device memory are accessible by
   pointers, but CUDA arrays are not.
*/
template <class Type, unsigned Dim>
class Pointer: virtual public Layout<Type, Dim>
{
public:
  typedef Iterator<Dim> iterator;

#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Pointer():
    buffer(0)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size
  */
  inline Pointer(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    buffer(0)
  {
  }

  /**
     Constructor.
     @param layout requested layout
  */
  inline Pointer(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    buffer(0)
  {
  }

  /**
     Destructor.
  */
  virtual ~Pointer()
  {
  }

  /**
     Array index operator.
     @param i index
     @return value at index i
  */
  inline Type &operator[](size_t i) { return buffer[i]; }

  /**
     Array index operator.
     @param i index
     @return value at index i
  */
  inline Type &operator[](const SizeBase<Dim> &i) { return buffer[this->getOffset(i)]; }

  /**
     Array index operator.
     @param i index
     @return value at index i (constant)
  */
  inline const Type &operator[](size_t i) const { return buffer[i]; }

  /**
     Array index operator.
     @param i index
     @return value at index i
  */
  inline const Type &operator[](const SizeBase<Dim> &i) const { return buffer[this->getOffset(i)]; }

  /**
     Get iterator for begin of data.
  */
  inline iterator begin() const { return iterator(this->size); }

  /**
     Get iterator for end of data.
  */
  inline iterator end() const { return iterator(this->size).setEnd(); }

  /**
     Get buffer pointer.
     @return buffer pointer (constant)
  */
  inline const Type *getBuffer() const { return buffer; }

  /**
     Get buffer pointer.
     @return buffer pointer
  */
  inline Type *getBuffer() { return buffer; }

protected:
  Type *buffer;

  inline Pointer(const Pointer<Type, Dim> &x):
    Layout<Type, Dim>(x),
    buffer(0)  // subclasses must perform a copy operation
  {
  }
};

}  // namespace Cuda


#endif
