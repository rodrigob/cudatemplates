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

#ifndef CUDA_DEVICEMEMORYPITCHED_H
#define CUDA_DEVICEMEMORYPITCHED_H


#include <cuda_runtime.h>

#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/staticassert.hpp>


namespace Cuda {

/**
   Representation of linear GPU memory with proper padding.
   Appropriate padding is added to maximize access performance.
*/
template <class Type, unsigned Dim>
class DeviceMemoryPitched:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public DeviceMemoryStorage<Type, Dim>
{
  CUDA_STATIC_ASSERT(Dim >= 2);

public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline DeviceMemoryPitched()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block
  */
  inline DeviceMemoryPitched(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemoryStorage<Type, Dim>(_size)
  {
    realloc();
  }

  /**
     Constructor.
     @param layout requested layout of memory block
  */
  inline DeviceMemoryPitched(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemoryStorage<Type, Dim>(layout)
  {
    realloc();
  }

#include "auto/copy_devicememorypitched.hpp"

  /**
     Allocate GPU memory.
  */
  void realloc();

  /**
     Allocate GPU memory.
     @_size size to be allocated
  */
  inline void realloc(const Size<Dim> &_size)
  {
    DeviceMemoryStorage<Type, Dim>::realloc(_size);
  }

  /**
     Initializes the GPU memory with the value \a val.
     Unfortunately only integer values are supported by the cudaMemset functions.
   */
  void initMem(int val, bool sync = true);
};

template <class Type, unsigned Dim>
void DeviceMemoryPitched<Type, Dim>::
realloc()
{
  CUDA_STATIC_ASSERT(Dim >= 2);
  this->free();

  if(Dim == 2) {
    // allocating empty data is not considered an error
    // since this is a normal operation within STL containers
    if((this->size[0] == 0) || (this->size[1] == 0)) {
      this->setPitch(0);
      return;
    }

    size_t pitch;
    CUDA_CHECK(cudaMallocPitch((void **)&this->buffer, &pitch, this->size[0] * sizeof(Type), this->size[1]));
    this->setPitch(pitch);
  }
  else if(Dim >= 3) {
    cudaExtent extent;
    extent.width = this->size[0] * sizeof(Type);
    extent.height = this->size[1];
    extent.depth = this->size[2];

    // map 4- and more-dimensional data sets to 3D data:
    for(unsigned i = 3; i < Dim; ++i)
      extent.depth *= this->size[i];

    // see comment above
    if((extent.width == 0) || (extent.height == 0) || (extent.depth == 0)) {
      this->setPitch(0);
      return;
    }

    cudaPitchedPtr pitchDevPtr;
    CUDA_CHECK(cudaMalloc3D(&pitchDevPtr, extent));
    this->buffer = (Type *)pitchDevPtr.ptr;
    this->setPitch(pitchDevPtr.pitch);
  }

  assert(this->buffer != 0);

#ifdef CUDA_DEBUG_INIT_MEMORY
  CUDA_CHECK(cudaMemset(this->buffer, 0, this->getBytes()));
#endif
}

template <class Type, unsigned Dim>
void DeviceMemoryPitched<Type, Dim>::
initMem(int val, bool sync)
{
  if(this->buffer == 0)
    return;

  if(Dim == 2) {
    CUDA_CHECK(cudaMemset2D(this->buffer, this->getPitch(), val, this->size[0] * sizeof(Type), this->size[1]));
  }
  else if(Dim >= 3) {
    cudaExtent extent;
    extent.width = this->getPitch();
    extent.height = this->size[1];
    extent.depth = this->size[2];

    cudaPitchedPtr pitchDevPtr;
    pitchDevPtr.ptr = (void *)this->buffer;
    pitchDevPtr.pitch = this->getPitch();
    pitchDevPtr.xsize = this->getPitch();
    pitchDevPtr.ysize = this->size[1];

    CUDA_CHECK(cudaMemset3D(pitchDevPtr, val, extent));
  }

  if(sync)
    cudaThreadSynchronize();
}

}  // namespace Cuda


#include "auto/specdim_devicememorypitched.hpp"


#endif
