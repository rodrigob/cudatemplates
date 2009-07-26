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


#ifdef __CUDACC__
#include <driver_types.h>
#include <texture_types.h>
#endif

#include <cudatemplates/error.hpp>
#include <cudatemplates/pointerstorage.hpp>


namespace Cuda {

template <class Type, unsigned Dim> class DeviceMemoryReference;

/**
   Representation of global GPU memory.
   This is the base class for all kind of GPU memory for which a pointer can be
   obtained for read and write access (i.e., no CUDA arrays).
*/
template <class Type, unsigned Dim>
class DeviceMemory:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>
{
public:
  typedef DeviceMemoryReference<Type, Dim> Reference;

#ifdef __CUDACC__
  typedef texture<Type, Dim, cudaReadModeElementType> Texture;
  typedef texture<Type, Dim, cudaReadModeNormalizedFloat> TextureNormalizedFloat;
#endif

  /**
     A stripped-down version of the layout data structure suitable for passing
     to a CUDA kernel.
  */
  struct KernelData
  {
    /**
       Data pointer.
    */
    Type *data;

    /**
       Data size.
    */
    size_t size[Dim];

    /**
       Data stride.
    */
    size_t stride[Dim];

    /**
       Default constructor.
    */
    KernelData(): data(0) {}

    /**
       Constructor.
       This constructor is invoked when a kernel expecting a KernelData
       argument is called with an instance of DeviceMemory, i.e., you don't
       have to construct the KernelData object explicitly.
       There is no distinction between const and non-const data.
       @param mem reference to device memory object
    */
    KernelData(const DeviceMemory<Type, Dim> &mem):
      data(const_cast<Type *>(mem.getBuffer()))
    {
      for(int i = Dim; i--;) {
        size[i] = mem.size[i];
        stride[i] = mem.stride[i];
      }
    }
  };

  /**
     Initializes the GPU memory with the value \a val.
     Unfortunately only integer values are supported by the cudaMemset functions.
  */
  void initMem(int val, bool sync = true);


  /**
    Returns a single slice from a higher dimensional dataset.
    Keeps region of interest and other information.
    @param slice slice to which reference will be created
  */
  DeviceMemoryReference<Type, Dim-1> getSlice(unsigned int slice)
  {
    CUDA_STATIC_ASSERT(Dim >= 2);

    if (slice>=this->size[Dim-1])
      CUDA_ERROR("out of bounds");

    // Calculate new size
    Cuda::Size<Dim-1> slice_size;
    for(int i = Dim-1; i--;)
      slice_size[i] = this->size[i];

    int offset = this->stride[Dim-2]*slice;
    DeviceMemoryReference<Type, Dim-1> slice_ref(slice_size, this->buffer + offset);

    for(int i = Dim-1; i--;)
    {
      slice_ref.region_ofs[i] = this->region_ofs[i];
      slice_ref.region_size[i] = this->region_size[i];
      slice_ref.stride[i] = this->stride[i];
      slice_ref.spacing[i] = this->spacing[i];
    }

    return slice_ref;
  }

#ifdef __CUDACC__

  template<enum cudaTextureReadMode readMode>
  void bindTexture(texture<Type, Dim, readMode> &tex) const
  {
    CUDA_STATIC_ASSERT(Dim <= 2);
    if (Dim == 1)
      {
	CUDA_CHECK(cudaBindTexture(0, tex, this->buffer, this->size[0] * sizeof(Type)));
      }
    else
      {
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Type>();
	CUDA_CHECK(cudaBindTexture2D(0, tex, this->buffer, channelDesc,
				     this->size[0], this->size[1], this->getPitch()));
      }
  }

  template<enum cudaTextureReadMode readMode>
  void unbindTexture(texture<Type, Dim, readMode> &tex) const
  {
    CUDA_STATIC_ASSERT(Dim <= 2);
    CUDA_CHECK(cudaUnbindTexture(tex));
  }

#endif

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

template <class Type, unsigned Dim>
void DeviceMemory<Type, Dim>::
initMem(int val, bool sync)
{
  if(this->buffer == 0)
    return;

  CUDA_CHECK(cudaMemset(this->buffer, val, this->getSize() * sizeof(Type)));

  if(sync)
    cudaThreadSynchronize();
}

/**
   Representation of global GPU memory managed by CUDA templates.
   This is the base class for all kind of GPU memory for which a pointer can be
   obtained for read and write access and for which memory management is
   performed by the CUDA templates and (i.e., no CUDA arrays).
*/
template <class Type, unsigned Dim>
class DeviceMemoryStorage:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public DeviceMemory<Type, Dim>,
    public PointerStorage<Type, Dim>
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

}  // namespace Cuda


#endif
