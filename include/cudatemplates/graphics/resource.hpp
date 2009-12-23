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

#ifndef CUDA_GRAPHICS_RESOURCE_H
#define CUDA_GRAPHICS_RESOURCE_H


#include <assert.h>

#include <cuda_runtime_api.h>

#include <cudatemplates/error.hpp>
#include <cudatemplates/layout.hpp>
#include <cudatemplates/opengl/error.hpp>


namespace Cuda {
namespace Graphics {

class Resource
{
public:
  inline Resource():
    resource(0), mapped(false)
  {
  }

  virtual ~Resource()
  {
    unregisterObject();
  }

  inline void map()
  {
    if(mapped)
      return;

    registerObject();
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource, 0));
    mapInternal();
    mapped = true;
  }

  virtual void registerObject() = 0;

  inline void setMapFlags(unsigned int flags)
  {
    CUDA_CHECK(cudaGraphicsResourceSetMapFlags(resource, flags));
  }

  inline void unmap()
  {
    if(!mapped)
      return;

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, 0));
    unmapInternal();
    mapped = false;
  }

  inline void unregisterObject()
  {
    if(resource == 0)
      return;

    unmap();
    CUDA_CHECK(cudaGraphicsUnregisterResource(resource));
    resource = 0;
  }

  /**
     compatibility methods (will be removed later):
  */
  inline void connect() { map(); }
  inline void disconnect() { unregisterObject(); }

protected:
  cudaGraphicsResource *resource;
  bool mapped;

private:
  virtual void mapInternal() = 0;
  virtual void unmapInternal() = 0;
};

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
    bufname(0), target(t), usage(u), flags(f)
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
    bufname(0), target(t), usage(u), flags(f)
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
    bufname(0), target(t), usage(u), flags(f)
  {
    realloc();
  }

  ~Buffer();

  // #include "auto/copy_opengl_bufferobject.hpp"

  /**
     Allocate buffer memory.
  */
  void realloc();

  /**
     Allocate buffer memory.
     @_size size to be allocated
  */
  inline void realloc(const Size<Dim> &_size)
  {
    DeviceMemoryStorage<Type, Dim>::realloc(_size);
  }

  /**
     Bind the buffer object to the target specified in the constructor.
  */
  inline void bind() { CUDA_OPENGL_CHECK(glBindBuffer(this->target, this->bufname)); }

  /**
     Bind the buffer object to the given target.
     @param t target to which the buffer object should be bound
  */
  inline void bind(GLenum t) { CUDA_OPENGL_CHECK(glBindBuffer(t, this->bufname)); }

  /**
     Register OpenGL buffer object for use in CUDA.
  */
  void registerObject()
  {
    if(resource != 0)
      return;

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&resource, bufname, flags));
    assert(resource != 0);
  }

  /**
     Unbind the buffer object from the target specified in the constructor.
  */
  inline void unbind() { CUDA_OPENGL_CHECK(glBindBuffer(this->target, 0)); }

  /**
     Unbind the buffer object from the given target.
     @param t target from which the buffer object should be unbound
  */
  inline static void unbind(GLenum t) {
    CUDA_OPENGL_CHECK(glBindBuffer(t, 0));
  }

  /**
     Free buffer memory.
  */
  void free();

private:
  /**
     Buffer object name.
  */
  GLuint bufname;

  /**
     Specifies the target to which the buffer object is bound.
  */
  GLenum target;

  /**
     Specifies the expected usage pattern of the data store.
  */
  GLenum usage;

  /**
     Flags for buffer registration in CUDA.
  */
  unsigned int flags;

  void mapInternal()
  {
    size_t bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&this->buffer, &bytes, resource));
    
    if(this->buffer == 0)
      CUDA_ERROR("map buffer object failed");

    if(bytes != this->getBytes())
      CUDA_ERROR("size mismatch");
  }

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

  CUDA_OPENGL_CHECK(glGenBuffers(1, &(this->bufname)));

  if(this->bufname == 0)
    CUDA_OPENGL_ERROR("generate buffer object failed");

  bind();
  CUDA_OPENGL_CHECK(glBufferData(this->target, p * sizeof(Type), 0, this->usage));
  unbind();
  map();
}

template <class Type, unsigned Dim>
void Buffer<Type, Dim>::
free()
{
  if(this->bufname == 0)
    return;

  unregisterObject();
  glBindBuffer(this->target, 0);
  glDeleteBuffers(1, &(this->bufname));
  bufname = 0;
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
