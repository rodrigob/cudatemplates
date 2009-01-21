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

#ifndef CUDA_POINTERSTORAGE_H
#define CUDA_POINTERSTORAGE_H


#include <cudatemplates/pointer.hpp>
#include <cudatemplates/storage.hpp>


namespace Cuda {

/**
   Class to represent memory that can be allocated and freed
   and is accessible via a typed pointer.
*/
template <class Type, unsigned Dim>
class PointerStorage:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public Storage<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline PointerStorage()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size
  */
  inline PointerStorage(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    Storage<Type, Dim>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout
  */
  inline PointerStorage(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    Storage<Type, Dim>(layout)
  {
  }

  /**
     Initialize data structure.
  */
  inline void init() { this->buffer = 0; }

protected:
  inline PointerStorage(const PointerStorage<Type, Dim> &x):
    Layout<Type, Dim>(x),
    Pointer<Type, Dim>(x),
    Storage<Type, Dim>(x)
  {
  }

};

}  // namespace Cuda


#endif
