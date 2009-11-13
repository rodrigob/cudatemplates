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

#ifndef CUDA_HOSTMEMORYHEAP_H
#define CUDA_HOSTMEMORYHEAP_H


#include <cudatemplates/error.hpp>
#include <cudatemplates/hostmemory.hpp>


namespace Cuda {

/**
   Representation of heap memory on the CPU.
*/
template <class Type, unsigned Dim>
class HostMemoryHeap:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public HostMemoryStorage<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline HostMemoryHeap()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block
  */
  inline HostMemoryHeap(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    HostMemoryStorage<Type, Dim>(_size)
  {
    realloc();
  }

  /**
     Constructor.
     @param layout requested layout of memory block
  */
  inline HostMemoryHeap(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    HostMemoryStorage<Type, Dim>(layout)
  {
    realloc();
  }

#include "auto/copy_hostmemoryheap.hpp"

  /**
     Constructor from different data type.
     @param x host memory data of different data type
  */
  template <class Type2>
  inline HostMemoryHeap(const HostMemoryHeap<Type2, Dim> &x):
    Layout<Type, Dim>(x.size),
    Pointer<Type, Dim>(x.size),
    HostMemoryStorage<Type, Dim>(x.size)
  {
    this->init();
    this->realloc();
    copy(*this, x);
  }

  /**
     Destructor.
  */
  ~HostMemoryHeap()
  {
    free();
  }

  /**
     Allocate CPU memory.
  */
  void realloc();

  /**
     Allocate CPU memory.
     @param _size requested size
  */
  inline void realloc(const Size<Dim> &_size)
  {
    HostMemoryStorage<Type, Dim>::realloc(_size);
  }

  /**
     Free CPU memory.
  */
  void free();
};

template <class Type, unsigned Dim>
void HostMemoryHeap<Type, Dim>::
realloc()
{
  this->setPitch(0);
  this->buffer = (Type *)malloc(this->getSize() * sizeof(Type));

  if(this->buffer == 0)
    CUDA_ERROR("out of memory");

#ifdef CUDA_DEBUG_INIT_MEMORY
  memset(this->buffer, 0, this->getBytes());
#endif
}

template <class Type, unsigned Dim>
void HostMemoryHeap<Type, Dim>::
free()
{
  if(this->buffer == 0)
    return;

  ::free(this->buffer);
  this->buffer = 0;
}

}  // namespace Cuda


#include "auto/specdim_hostmemoryheap.hpp"


#endif
