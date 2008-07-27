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

#ifndef CUDA_DEVICEMEMORYLINEAR_H
#define CUDA_DEVICEMEMORYLINEAR_H


#include <cuda_runtime.h>

#include <cudatemplates/devicememory.hpp>


#define CUDA_INIT_POINTER(...) __VA_ARGS__,


namespace Cuda {

/**
   Representation of linear GPU memory.
   No padding is performed, i.e., efficiency of access may be suboptimal.
*/
template <class Type, unsigned Dim>
class DeviceMemoryLinear: public DeviceMemoryStorage<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline DeviceMemoryLinear()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
  */
  inline DeviceMemoryLinear(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemoryStorage<Type, Dim>(_size)
  {
    alloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
  */
  inline DeviceMemoryLinear(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemoryStorage<Type, Dim>(layout)
  {
    alloc();
  }

  CUDA_COPY_CONSTRUCTOR(DeviceMemoryLinear, DeviceMemoryStorage)

  /**
     Allocate GPU memory.
  */
  void alloc();
};

template <class Type, unsigned Dim>
void DeviceMemoryLinear<Type, Dim>::
alloc()
{
  this->free();
  size_t p = 1;

  for(int i = Dim; i--;)
    p *= this->size[i];

  CUDA_CHECK(cudaMalloc((void **)&this->buffer, p * sizeof(Type)));
  this->setPitch(0);
  assert(this->buffer != 0);
}

CUDA_SPECIALIZE_DIM(DeviceMemoryLinear)

}


#undef CUDA_INIT_POINTER


#endif
