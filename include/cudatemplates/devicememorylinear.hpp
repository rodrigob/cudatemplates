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


namespace Cuda {

/**
   Representation of linear GPU memory.
   No padding is performed, i.e., efficiency of access may be suboptimal.
*/
template <class Type, unsigned Dim>
class DeviceMemoryLinear:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public DeviceMemoryStorage<Type, Dim>
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
    realloc();
  }

  /**
     Constructor.
     @param layout requested layout of memory block.
  */
  inline DeviceMemoryLinear(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemoryStorage<Type, Dim>(layout)
  {
    realloc();
  }

#include "auto/copy_devicememorylinear.hpp"

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
};

template <class Type, unsigned Dim>
void DeviceMemoryLinear<Type, Dim>::
realloc()
{
  this->free();
  size_t p = 1;

  for(size_t i = Dim; i--;)
    p *= this->size[i];

  // allocating empty data is not considered an error
  // since this is a normal operation within STL containers
  if(p == 0) {
    this->setPitch(0);
    return;
  }

  CUDA_CHECK(cudaMalloc((void **)&this->buffer, p * sizeof(Type)));
  this->setPitch(0);

  if(this->buffer == 0)
    CUDA_ERROR("cudaMalloc failed");

#ifdef CUDA_DEBUG_INIT_MEMORY
  CUDA_CHECK(cudaMemset(this->buffer, 0, this->getBytes()));
#endif
}

}  // namespace Cuda


#include "auto/specdim_devicememorylinear.hpp"


#endif
