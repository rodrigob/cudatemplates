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

#ifndef CUDA_GRAPHICS_OPENGL_BUFFER_H
#define CUDA_GRAPHICS_OPENGL_BUFFER_H


#include <assert.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/error.hpp>
#include <cudatemplates/layout.hpp>
#include <cudatemplates/opengl/error.hpp>
#include <cudatemplates/graphics/opengl/resource.hpp>


namespace Cuda {
namespace Graphics {
namespace OpenGL {

template <class Type, unsigned Dim>
class Buffer: public DeviceMemoryStorage<Type, Dim>, public Resource
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline Buffer(GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW, unsigned int f = 0):
    Resource(f),
    target(t), usage(u)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline Buffer(const Size<Dim> &_size, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW, unsigned int f = 0):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemoryStorage<Type, Dim>(_size),
    Resource(f),
    target(t), usage(u)
  {
    realloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline Buffer(const Layout<Type, Dim> &layout, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW, unsigned int f = 0):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemoryStorage<Type, Dim>(layout),
    Resource(f),
    target(t), usage(u)
  {
    realloc();
  }

  ~Buffer();

  // #include "auto/copy_opengl_bufferobject.hpp"

  /**
     Bind the buffer object to the given target.
     @param t target to which the buffer object should be bound
  */
  /*
  inline void bind(GLenum t)
  {
    CUDA_OPENGL_CHECK(glBindBuffer(t, this->name));
  }
  */

  /**
     Free buffer memory.
  */
  void free();

  /**
     Allocate buffer memory.
  */
  void realloc();

  /**
     Allocate buffer memory.
     @_size size to be allocated
  */
  /*
  inline void realloc(const Size<Dim> &_size)
  {
    DeviceMemoryStorage<Type, Dim>::realloc(_size);
  }
  */

  /**
     Set buffer object target.
     If the buffer object is currently bound to a different target, it will be
     unbound and then bound to the new target.
     @param target_new new target
     @return old target
  */
  GLenum setTarget(GLenum target_new)
  {
    if(target_new == target)
      return target;

    GLenum target_old = target;
    state_t state = getState();

    if(state == STATE_GRAPHICS_BOUND)
      setState(STATE_INACTIVE);

    target = target_new;
    setState(state);
    return target_old;
  }

  /**
     Unbind the buffer object from the given target.
     @param t target from which the buffer object should be unbound
  */
  /*
  inline static void unbind(GLenum t) {
    CUDA_OPENGL_CHECK(glBindBuffer(t, 0));
  }
  */

#ifdef CUDA_GRAPHICS_COMPATIBILITY
  inline void bind() { Graphics::Resource::bind(); }
#endif

private:
  /**
     Specifies the target to which the buffer object is bound.
  */
  GLenum target;

  /**
     Specifies the expected usage pattern of the data store.
  */
  GLenum usage;

  /**
     Bind the buffer object to the target specified in the constructor.
  */
  void bindObjectInternal()
  {
    CUDA_OPENGL_CHECK(glBindBuffer(this->target, this->name));
  }

  /**
     Check map status.
     @return true if buffer is currently mapped for use with CUDA
  */
  bool isMapped() const
  {
    return this->buffer != 0;
  }

  /**
     Object-specific part of map action.
  */
  void mapInternal()
  {
    size_t bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&this->buffer, &bytes, resource));
    
    if(this->buffer == 0)
      CUDA_ERROR("map buffer object failed");

    if(bytes != this->getBytes())
      CUDA_ERROR("size mismatch");
  }

  /**
     Register OpenGL buffer object for use with CUDA.
  */
  void registerObject()
  {
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&resource, name, flags));

    if(this->resource == 0)
      CUDA_ERROR("register buffer object failed");
  }

  /**
     Unbind the buffer object from the target specified in the constructor.
  */
  inline void unbindObjectInternal()
  {
    CUDA_OPENGL_CHECK(glBindBuffer(this->target, 0));
  }

  /**
     Object-specific part of unmap action.
  */
  void unmapInternal()
  {
    this->buffer = 0;
  }
};

template <class Type, unsigned Dim>
void Buffer<Type, Dim>::
realloc()
{
  this->free();
  this->setPitch(0);
  size_t p = 1;

  for(size_t i = Dim; i--;)
    p *= this->size[i];

  CUDA_OPENGL_CHECK(glGenBuffers(1, &(this->name)));

  if(this->name == 0)
    CUDA_ERROR("generate buffer object failed");

  // do an explicit bind/unbind since the state "INACTIVE" can only be entered
  // after data has been allocated with glBufferData:
  bindObjectInternal();
  CUDA_OPENGL_CHECK(glBufferData(this->target, p * sizeof(Type), 0, this->usage));
  unbindObjectInternal();

  // prepare object for use with CUDA:
  setState(STATE_CUDA_MAPPED);
}

template <class Type, unsigned Dim>
void Buffer<Type, Dim>::
free()
{
  if(this->name == 0)
    return;

  setState(STATE_UNUSED);
  CUDA_OPENGL_CHECK(glDeleteBuffers(1, &(this->name)));
  name = 0;
}

template <class Type, unsigned Dim>
Buffer<Type, Dim>::
~Buffer()
{
  this->free();
}

}  // namespace OpenGL
}  // namespace Graphics
}  // namespace Cuda


#endif
