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

#ifndef CUDA_DEVICEMEMORYREFERENCE_H
#define CUDA_DEVICEMEMORYREFERENCE_H


#include <cudatemplates/devicememory.hpp>


namespace Cuda {

/**
   Reference to existing buffer in GPU main memory.
   This class can be used to apply the CUDA Templates methods to memory regions
   managed by other libraries.
*/
template <class Type, unsigned Dim>
class DeviceMemoryReference: public DeviceMemory<Type, Dim>
{
public:
  /**
     Constructor.
     @param requested layout
     @param _buffer pointer to GPU memory
  */
  inline DeviceMemoryReference(const Layout<Type, Dim> &layout, Type *_buffer):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemory<Type, Dim>(layout)
  {
    this->buffer = _buffer;
  }

  /**
     Constructor.
     The current implementation only works if the resulting object refers to a
     contiguous block of memory.
     @param requested layout
     @param _buffer pointer to GPU memory
  */
  inline DeviceMemoryReference(DeviceMemory<Type, Dim> &data, const Size<Dim> &ofs, const Size<Dim> &_size):
    Layout<Type, Dim>(data),
    Pointer<Type, Dim>(data),
    DeviceMemory<Type, Dim>(data)
  {
    this->buffer = data.getBuffer() + data.getOffset(ofs);
    this->size = _size;
    this->setPitch(0);
  }

protected:
  /**
     Default constructor.
     This is only for subclasses which know how to correctly set up the data
     pointer.
  */
  inline DeviceMemoryReference() {}

};

}


#include "auto/specdim_devicememoryreference.hpp"


#endif
