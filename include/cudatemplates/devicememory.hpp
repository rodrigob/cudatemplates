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

#ifndef CUDA_DEVICEMEMORY_H
#define CUDA_DEVICEMEMORY_H


#include <cudatemplates/error.hpp>
#include <cudatemplates/storage.hpp>


namespace Cuda {

template <class Type, unsigned Dim> class DeviceMemoryReference;

/**
   Representation of GPU memory.
   This is the base class for all kind of GPU memory except CUDA arrays.
*/
template <class Type, unsigned Dim>
class DeviceMemory: virtual public Pointer<Type, Dim>
{
public:
  typedef DeviceMemoryReference<Type, Dim> Reference;

protected:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline DeviceMemory()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block
  */
  inline DeviceMemory(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout of memory block
  */
  inline DeviceMemory(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout)
  {
  }

  inline DeviceMemory(const DeviceMemory<Type, Dim> &x):
    Layout<Type, Dim>(x),
    Pointer<Type, Dim>(x)
  {
  }
};

/**
   Representation of GPU memory managed by the CUDA Toolkit.
   This is the base class for all kind of GPU memory for which memory
   management is performed by the CUDA Toolkit (except CUDA arrays).
*/
template <class Type, unsigned Dim>
class DeviceMemoryStorage: public DeviceMemory<Type, Dim>, public PointerStorage<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline DeviceMemoryStorage()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block
  */
  inline DeviceMemoryStorage(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemory<Type, Dim>(_size),
    PointerStorage<Type, Dim>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout of memory block
  */
  inline DeviceMemoryStorage(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemory<Type, Dim>(layout),
    PointerStorage<Type, Dim>(layout)
  {
  }

  ~DeviceMemoryStorage();

  void free();
  inline void init() { this->buffer = 0; }

protected:
  inline DeviceMemoryStorage(const DeviceMemoryStorage<Type, Dim> &x):
    Layout<Type, Dim>(x),
    Pointer<Type, Dim>(x),
    DeviceMemory<Type, Dim>(x),
    PointerStorage<Type, Dim>(x)
  {
  }
};

template <class Type, unsigned Dim>
void DeviceMemoryStorage<Type, Dim>::
free()
{
  if(this->buffer == 0)
    return;

  CUDA_CHECK(cudaFree(this->buffer));
  this->buffer = 0;
}

template <class Type, unsigned Dim>
DeviceMemoryStorage<Type, Dim>::
~DeviceMemoryStorage()
{
  this->free();
}

}


#endif
