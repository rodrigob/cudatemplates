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
class DeviceMemoryReference:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public DeviceMemory<Type, Dim>
{
public:
  /**
     Constructor.
     @param _size requested size
     @param _buffer pointer to GPU memory
  */
  inline DeviceMemoryReference(const Size<Dim> &_size, Type *_buffer):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemory<Type, Dim>(_size)
  {
    this->buffer = _buffer;
  }

  /**
     Constructor.
     @param layout requested layout
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
     Constructor based on existing device memory. Will keep any region of interest
     valid, by determining 'intersection' of regions.
     @param data existing device memroy
     @param ofs offset to new region
     @param _size size of new region
  */
  inline DeviceMemoryReference( DeviceMemory<Type, Dim> &data, const Size<Dim> &ofs,
                                const Size<Dim> &_size):
    Layout<Type, Dim>(data),
    Pointer<Type, Dim>(data),
    DeviceMemory<Type, Dim>(data)
  {
    this->buffer = data.getBuffer() + data.getOffset(ofs);
    this->size = _size;

    for(int i = Dim; i--;)
    {
      this->region_ofs[i] = std::min(std::max((int)this->region_ofs[i]-(int)ofs[i], 0), (int)this->size[i]);
      this->region_size[i] = std::min(this->region_size[i], this->size[i] - this->region_ofs[i]);
    }
  }

protected:
  /**
     Default constructor.
     This is only for subclasses which know how to correctly set up the data
     pointer.
  */
  inline DeviceMemoryReference() {}

};

}  // namespace Cuda


#include "auto/specdim_devicememoryreference.hpp"


#endif
