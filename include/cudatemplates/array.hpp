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

#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H


#ifdef __CUDACC__
#include <driver_types.h>
#include <texture_types.h>
#endif

#include <cudatemplates/error.hpp>
#include <cudatemplates/staticassert.hpp>
#include <cudatemplates/storage.hpp>


namespace Cuda {

/**
   Representation of CUDA array.
   CUDA arrays allow access to texture hardware.
*/
template <class Type, unsigned Dim>
class Array:
    virtual public Layout<Type, Dim>,
    public Storage<Type, Dim>
{
public:

#ifdef __CUDACC__
  typedef texture<Type, Dim, cudaReadModeElementType> Texture;
  typedef texture<Type, Dim, cudaReadModeNormalizedFloat> TextureNormalizedFloat;
#endif

#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Array():
    array(0)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of array
  */
  inline Array(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Storage<Type, Dim>(_size),
    array(0)
  {
    realloc();
  }

  /**
     Constructor.
     @param layout requested layout of array
  */
  inline Array(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Storage<Type, Dim>(layout),
    array(0)
  {
    realloc();
  }

#include "auto/copy_array.hpp"

  /**
     Destructor.
  */
  ~Array()
  {
    free();
  }

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
    Storage<Type, Dim>::realloc(_size);
  }

#ifdef __CUDACC__

  template<enum cudaTextureReadMode readMode>
  void bindTexture(texture<Type, Dim, readMode> &tex) const
  {
    CUDA_CHECK(cudaBindTextureToArray(tex, array));
  }

  template<enum cudaTextureReadMode readMode>
  void unbindTexture(texture<Type, Dim, readMode> &tex) const
  {
    CUDA_CHECK(cudaUnbindTexture(tex));
  }

#endif

  /**
     Free GPU memory.
  */
  void free();

  /**
     Get pointer to CUDA array structure.
     @return pointer to cudaArray
  */
  inline cudaArray *getArray() { return array; }

  /**
     Get pointer to CUDA array structure.
     @return pointer to cudaArray
  */
  inline const cudaArray *getArray() const { return array; }
  
  /**
     Initialize CUDA array pointer.
  */
  inline void init() { array = 0; }

protected:
  cudaArray *array;
};

template <class Type, unsigned Dim>
void Array<Type, Dim>::
realloc()
{
  CUDA_STATIC_ASSERT(Dim >= 1);
  this->free();

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Type>();

  if(Dim == 1) {
    CUDA_CHECK(cudaMallocArray(&array, &channelDesc, this->size[0], 1));
  }
  else if(Dim == 2) {
    CUDA_CHECK(cudaMallocArray(&array, &channelDesc, this->size[0], this->size[1]));
  }
  else {
    cudaExtent extent;
    extent.width = this->size[0];
    extent.height = this->size[1];
    extent.depth = this->size[2];

    // map 4- and more-dimensional data sets to 3D data:
    for(unsigned i = 3; i < Dim; ++i)
      extent.depth *= this->size[i];

    CUDA_CHECK(cudaMalloc3DArray(&array, &channelDesc, extent));
  }
}

template <class Type, unsigned Dim>
void Array<Type, Dim>::
free()
{
  if(array == 0)
    return;
  
  CUDA_CHECK(cudaFreeArray(array));
  array = 0;
}

}


#include "auto/specdim_array.hpp"


#endif
