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

#ifndef CUDA_DEVICEMEMORYREFERENCE1D_H
#define CUDA_DEVICEMEMORYREFERENCE1D_H


#include <cudatemplates/devicememoryreference.hpp>


namespace Cuda {

/**
   DeviceMemoryReference template specialized for 1 dimension(s).
 */
template <class Type>
class DeviceMemoryReference1D: public DeviceMemoryReference<Type, 1>
{
public:
  inline DeviceMemoryReference1D()
  {
  }

  inline DeviceMemoryReference1D(const Size<1> &_size, Type *_buffer):
    Layout<Type, 1>(_size),
    Pointer<Type, 1>(_size),
    DeviceMemoryReference<Type, 1>(_size, _buffer)
  {
  }

  inline DeviceMemoryReference1D(const Layout<Type, 1> &layout, Type *_buffer):
    Layout<Type, 1>(layout),
    Pointer<Type, 1>(layout),
    DeviceMemoryReference<Type, 1>(layout, _buffer)
  {
  }

  inline DeviceMemoryReference1D(size_t size0, Type *_buffer):
    Layout<Type, 1>(Size<1>(size0)),
    Pointer<Type, 1>(Size<1>(size0)),
    DeviceMemoryReference<Type, 1>(Size<1>(size0), _buffer)
  {
  }

  inline void alloc()
  {
    DeviceMemoryReference<Type, 1>::alloc();
  }

  inline void alloc(size_t size0)
  {
    Storage<Type, 1>::alloc(Size<1>(size0));
  }
};

}  // namespace Cuda


#endif
