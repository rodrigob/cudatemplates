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

#ifndef CUDA_HOSTMEMORYREFERENCE_H
#define CUDA_HOSTMEMORYREFERENCE_H

#include <cudatemplates/hostmemory.hpp>

namespace Cuda {

/**
   Reference to existing buffer in CPU main memory.
   This class can be used to apply the Cuda Templates methods to memory regions
   managed by other libraries.
*/
template <class Type, unsigned Dim>
class HostMemoryReference:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public HostMemory<Type, Dim>
{
public:
  /**
     Constructor.
     @param _size requested size
     @param _buffer pointer to CPU memory
  */
  inline HostMemoryReference(const Size<Dim> &_size, Type *_buffer):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    HostMemory<Type, Dim>(_size)
  {
    this->buffer = _buffer;
  }

  /**
     Constructor.
     @param layout requested layout
     @param _buffer pointer to CPU memory
  */
  inline HostMemoryReference(const Layout<Type, Dim> &layout, Type *_buffer):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    HostMemory<Type, Dim>(layout)
  {
    this->buffer = _buffer;
  }

  /**
    Constructor based on existing host memory. Will keep any region of interest
    valid, by determining 'intersection' of regions.
    @param data existing device memroy
    @param ofs offset to new region
    @param _size size of new region
  */
  inline HostMemoryReference( HostMemory<Type, Dim> &data, const Size<Dim> &ofs,
                              const Size<Dim> &_size):
    Layout<Type, Dim>(data),
    Pointer<Type, Dim>(data),
    HostMemory<Type, Dim>(data)
  {
    this->buffer = data.getBuffer() + data.getOffset(ofs);
    this->size = _size;

    for(int i = Dim; i--;)
    {
      // Min and Max did not work here !!!
      int new_ofs = (int)this->region_ofs[i]-(int)ofs[i];
      if (new_ofs < 0)
        new_ofs = 0;
      if (new_ofs > (int)this->size[i])
        new_ofs = (int)this->size[i];
      this->region_ofs[i] = new_ofs;
      int new_size = this->region_size[i];
      if (this->region_size[i] > this->size[i] - this->region_ofs[i])
        new_size = this->size[i] - this->region_ofs[i];
      this->region_size[i] = new_size;
    }
  }

 /**
    Copy Constructor.
    Ensures that the layout and pointer is correct when the copy constructor is invoked.
    @param other existing device memory reference that is copied.
 */
  inline HostMemoryReference(const HostMemoryReference<Type, Dim> &other) :
    Layout<Type, Dim>(other),
    Pointer<Type, Dim>(other),
    HostMemory<Type, Dim>(other)
  {
    this->buffer = other.buffer;
    this->size = other.size;
    this->region_ofs = other.region_ofs;
    this->region_size = other.region_size;
    this->stride = other.stride;
    for(size_t i = 0; i < Dim; ++i)
    {
      this->spacing[i] = other.spacing[i];
    }
  }

  /**
     Asignment operator.
     Ensures that the layout and pointer is correct when the asignment operator is invoked.
     @param other existing device memory reference that is copied.
  */
  inline HostMemoryReference& operator= (const HostMemoryReference& other)
  {
    this->buffer = other.buffer;
    this->size = other.size;
    this->region_ofs = other.region_ofs;
    this->region_size = other.region_size;
    this->stride = other.stride;
    for(size_t i = 0; i < Dim; ++i)
    {
      this->spacing[i] = other.spacing[i];
    }
    return *this;
  }


protected:
  /**
     Default constructor.
     This is only for subclasses which know how to correctly set up the data
     pointer.
  */
  inline HostMemoryReference() {}

};

}  // namespace Cuda


#include "auto/specdim_hostmemoryreference.hpp"


#endif
