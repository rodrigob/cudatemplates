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

#ifndef CUDA_GRAPHICS_OPENGL_IMAGE_H
#define CUDA_GRAPHICS_OPENGL_IMAGE_H

/*
#include <assert.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <cudatemplates/error.hpp>
#include <cudatemplates/layout.hpp>
#include <cudatemplates/opengl/error.hpp>
*/
#include <cudatemplates/array.hpp>
#include <cudatemplates/graphics/resource.hpp>


namespace Cuda {
namespace Graphics {
namespace OpenGL {

template <class Type, unsigned Dim>
class Image:
    virtual public Layout<Type, Dim>,
    public Array<Type, Dim>,
    public Resource
{
protected:
  /**
     Constructor.
     @param t target to which the image object is bound
     @param u usage pattern of the data store
  */
  inline Image(unsigned int f):
    Layout<Type, Dim>(),
    Array<Type, Dim>(),
    Resource(f)
  {
  }

  /**
     Constructor.
     @param _size requested size of memory block.
     @param u usage pattern of the data store
  */
  inline Image(const Size<Dim> &_size, unsigned int f):
    Layout<Type, Dim>(_size),
    Array<Type, Dim>(),
    Resource(f)
  {
    // realloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
     @param u usage pattern of the data store
  */
  inline Image(const Layout<Type, Dim> &layout, unsigned int f):
    Layout<Type, Dim>(layout),
    Array<Type, Dim>(),
    Resource(f)
  {
    // realloc();
  }

private:
  /**
     Check map status.
     @return true if image is currently mapped for use with CUDA
  */
  bool isMapped() const
  {
    return this->array != 0;
  }

  /**
     Object-specific part of map action.
  */
  void mapInternal()
  {
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&this->array, resource, 0, 0));

    if(this->array == 0)
      CUDA_ERROR("map image object failed");
  }

  /**
     Object-specific part of unmap action.
  */
  void unmapInternal()
  {
    this->array = 0;
  }
};

}  // namespace OpenGL
}  // namespace Graphics
}  // namespace Cuda


#endif
