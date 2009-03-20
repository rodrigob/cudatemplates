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

#ifndef CUDA_DEVICEMEMORYMAPPED_H
#define CUDA_DEVICEMEMORYMAPPED_H


#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/hostmemorylocked.hpp>


namespace Cuda {

/**
   Mapped to existing buffer in GPU main memory.
   This class can be used to apply the CUDA Templates methods to memory regions
   managed by other libraries.
*/
template <class Type, unsigned Dim>
class DeviceMemoryMapped:
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
  inline DeviceMemoryMapped(const Size<Dim> &_size, Type *hbuffer):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemory<Type, Dim>(_size)
  {
    getDevicePointer(hbuffer);
  }

  /**
     Constructor.
     @param layout requested layout
     @param _buffer pointer to GPU memory
  */
  inline DeviceMemoryMapped(const Layout<Type, Dim> &layout, Type *hbuffer):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemory<Type, Dim>(layout)
  {
    getDevicePointer(hbuffer);
  }

  /**
     Constructor based on existing device memory. Will keep any region of interest
     valid, by determining 'intersection' of regions.
     @param data existing device memroy
  */
  inline DeviceMemoryMapped(HostMemoryLocked<Type, Dim> &data):
    Layout<Type, Dim>(data),
    Pointer<Type, Dim>(data),
    DeviceMemory<Type, Dim>(data)
  {
    getDevicePointer(data.getBuffer());
  }

protected:
  /**
     Default constructor.
     This is only for subclasses which know how to correctly set up the data
     pointer.
  */
  inline DeviceMemoryMapped() {}

private:
  inline void getDevicePointer(Type *hbuffer)
  {
    unsigned flags = 0;  // no documentation available
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&this->buffer, hbuffer, flags));
    assert(this->buffer != 0);
  }
};

}  // namespace Cuda


// #include "auto/specdim_devicememorymapped.hpp"


#endif
