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

#ifndef CUDA_OPENGL_BUFFEROBJECT_H
#define CUDA_OPENGL_BUFFEROBJECT_H


#include <GL/gl.h>

#ifndef _WIN32
#include <GL/glext.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/error.hpp>
#include <cudatemplates/opengl/error.hpp>
#include <cudatemplates/opengl/type.hpp>


namespace Cuda {
namespace OpenGL {

/**
   Representation of OpenGL buffer object.
*/
template <class Type, unsigned Dim>
class BufferObject: public DeviceMemoryStorage<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline BufferObject(GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW):
    bufname(0), target(t), usage(u), registered(false)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline BufferObject(const Size<Dim> &_size, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemoryStorage<Type, Dim>(_size),
    bufname(0), target(t), usage(u), registered(false)
  {
    realloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline BufferObject(const Layout<Type, Dim> &layout, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemoryStorage<Type, Dim>(layout),
    bufname(0), target(t), usage(u),
    registered(false)
  {
    realloc();
  }

  ~BufferObject();

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
     Register and map buffer object.
     If you called disconnect(), this must be called before using the buffer
     memory in a CUDA kernel.
  */
  void connect();

  /**
     Unmap and unregister buffer object.
     This must be called before accessing the buffer memory in OpenGL, e.g.,
     copying the buffer data from or to a texture.
  */
  void disconnect();

  /**
     Register abuffer object.
     If you called disconnect() or unregister, this must be called before using
     the buffer memory in a CUDA kernel.
  */
  void registerObject();

  /**
     Unregister buffer object.
     This must be called before accessing the buffer memory in OpenGL for
     writing, e.g., copying the buffer data from a texture.
  */
  void unregisterObject();

  /**
     Map buffer object.
     This must be called before accessing buffer memory in Cuda.
     Note that the buffer object must be registered in Cuda
  */
  void mapBuffer();

  /**
     Unmap buffer object.
     This must be called before read-accessing buffer memory in OpenGL.
     Note that the buffer object must be registered in Cuda.
  */
  void unmapBuffer();

  /**
     Free buffer memory.
  */
  void free();

  inline GLuint getName() const { return this->bufname; }

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
      Specifies whether a buffer is registered in Cuda
  */
  bool registered;
};

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
connect()
{
  registerObject();
  mapBuffer();
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
disconnect()
{
  if(registered)
    unmapBuffer();
  unregisterObject();
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
registerObject()
{
  if (!registered)
    {
      CUDA_CHECK(cudaGLRegisterBufferObject(this->bufname));
      registered = true;
    }
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
unregisterObject()
{
  if (registered)
    {
      CUDA_CHECK(cudaGLUnregisterBufferObject(this->bufname));
      registered = false;
    }
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
mapBuffer()
{
  if(this->buffer != 0)
    return;

  if (!registered) CUDA_ERROR("map buffer object failed - register it first");
  CUDA_CHECK(cudaGLMapBufferObject((void **)&this->buffer, this->bufname));

  if(this->buffer == 0)
    CUDA_ERROR("map buffer object failed");
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
unmapBuffer()
{
  if(this->buffer == 0)
    return;
  if (!registered) CUDA_ERROR("unmap of unregistered buffer object failed");
  CUDA_CHECK(cudaGLUnmapBufferObject(this->bufname));
  this->buffer = 0;
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
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
  connect();
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
free()
{
  if(this->bufname == 0)
    return;

  disconnect();
  glBindBuffer(this->target, 0);
  glDeleteBuffers(1, &(this->bufname));
  bufname = 0;
}

template <class Type, unsigned Dim>
BufferObject<Type, Dim>::
~BufferObject()
{
  this->free();
}

}  // namespace OpenGL
}  // namespace Cuda


#include "../auto/specdim_bufferobject.hpp"


#endif
