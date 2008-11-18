/*
  NOTE: THIS FILE HAS BEEN CREATED AUTOMATICALLY,
  ANY CHANGES WILL BE OVERWRITTEN WITHOUT NOTICE!
*/

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

#ifndef CUDA_DEVICEMEMORYREFERENCE2D_H
#define CUDA_DEVICEMEMORYREFERENCE2D_H


#include <cudatemplates/devicememoryreference.hpp>


namespace Cuda {

/**
   DeviceMemoryReference template specialized for 2 dimension(s).
 */
template <class Type>
class DeviceMemoryReference2D: public DeviceMemoryReference<Type, 2>
{
public:
  inline DeviceMemoryReference2D()
  {
  }

  inline DeviceMemoryReference2D(const Size<2> &_size, Type *_buffer):
    Layout<Type, 2>(_size),
    Pointer<Type, 2>(_size),
    DeviceMemoryReference<Type, 2>(_size, _buffer)
  {
  }

  inline DeviceMemoryReference2D(const Layout<Type, 2> &layout, Type *_buffer):
    Layout<Type, 2>(layout),
    Pointer<Type, 2>(layout),
    DeviceMemoryReference<Type, 2>(layout, _buffer)
  {
  }

  inline DeviceMemoryReference2D(size_t size0, size_t size1, Type *_buffer):
    Layout<Type, 2>(Size<2>(size0, size1)),
    Pointer<Type, 2>(Size<2>(size0, size1)),
    DeviceMemoryReference<Type, 2>(Size<2>(size0, size1), _buffer)
  {
  }

  inline void alloc()
  {
    DeviceMemoryReference<Type, 2>::alloc();
  }

  inline void alloc(size_t size0, size_t size1)
  {
    Storage<Type, 2>::alloc(Size<2>(size0, size1));
  }
};

}  // namespace Cuda


#endif
