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

#ifndef CUDA_HOSTMEMORY_H
#define CUDA_HOSTMEMORY_H


#include <cstdlib>
#include <cstring>

#include <cudatemplates/error.hpp>
#include <cudatemplates/pointerstorage.hpp>


namespace Cuda {

template <class Type, unsigned Dim> class HostMemoryReference;

/**
   Representation of CPU memory.
   This is the base class for all kind of CPU memory.
*/
template <class Type, unsigned Dim>
class HostMemory:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>
{
public:
  typedef HostMemoryReference<Type, Dim> Reference;

  /**
  Returns a single slice from a higher dimensional dataset.
  Keeps region of interest and other information.
  @param slice slice to which reference will be created
  */
  HostMemoryReference<Type, Dim-1> getSlice(unsigned int slice)
  {
    CUDA_STATIC_ASSERT(Dim >= 2);

    if (slice>=this->size[Dim-1])
      CUDA_ERROR("out of bounds");

    // Calculate new size
    Cuda::Size<Dim-1> slice_size;
    for(int i = Dim-1; i--;)
      slice_size[i] = this->size[i];

    int offset = this->stride[Dim-2]*slice;
    HostMemoryReference<Type, Dim-1> slice_ref(slice_size, this->buffer + offset);

    for(int i = Dim-1; i--;)
    {
      slice_ref.region_ofs[i] = this->region_ofs[i];
      slice_ref.region_size[i] = this->region_size[i];
      slice_ref.stride[i] = this->stride[i];
      slice_ref.spacing[i] = this->spacing[i];
    }

    return slice_ref;
  }

protected:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline HostMemory()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block
  */
  inline HostMemory(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout of memory block
  */
  inline HostMemory(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout)
  {
  }

  inline HostMemory(const HostMemory<Type, Dim> &x):
    Layout<Type, Dim>(x),
    Pointer<Type, Dim>(x)
  {
  }
};

/**
   Representation of CPU memory managed by the CUDA Toolkit.
   This is the base class for all kind of CPU memory for which memory
   management is performed by the CUDA Toolkit.
*/
template <class Type, unsigned Dim>
class HostMemoryStorage:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public HostMemory<Type, Dim>,
    public PointerStorage<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline HostMemoryStorage()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block
  */
  inline HostMemoryStorage(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    HostMemory<Type, Dim>(_size),
    PointerStorage<Type, Dim>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout of memory block
  */
  inline HostMemoryStorage(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    HostMemory<Type, Dim>(layout),
    PointerStorage<Type, Dim>(layout)
  {
  }

  /**
     Constructor from different data type.
     Don't try to replicate layout, just use size.
     @param x host memory data of different data type
  */
  template <class Type2>
  inline HostMemoryStorage(const HostMemory<Type2, Dim> &x):
    Layout<Type, Dim>(x.size),
    Pointer<Type, Dim>(x.size),
    HostMemory<Type, Dim>(x.size),
    PointerStorage<Type, Dim>(x.size)
  {
  }

  inline void init() { this->buffer = 0; }

protected:
  inline HostMemoryStorage(const HostMemoryStorage<Type, Dim> &x):
    Layout<Type, Dim>(x),
    Pointer<Type, Dim>(x),
    HostMemory<Type, Dim>(x),
    PointerStorage<Type, Dim>(x)
  {
  }
};

}  // namespace Cuda


#endif
