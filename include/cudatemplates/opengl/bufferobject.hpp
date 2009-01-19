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
  */
  inline BufferObject():
    bufname(0), target(GL_ARRAY_BUFFER), usage(GL_STATIC_DRAW)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
  */
  inline BufferObject(const Size<Dim> &_size, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemoryStorage<Type, Dim>(_size),
    bufname(0), target(t), usage(u)
  {
    alloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
  */
  inline BufferObject(const Layout<Type, Dim> &layout, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemoryStorage<Type, Dim>(layout),
    bufname(0), target(t), usage(u)
  {
    alloc();
  }

  ~BufferObject();

  // #include "auto/copy_opengl_bufferobject.hpp"

  /**
     Allocate buffer memory.
  */
  void alloc();

  /**
     Allocate buffer memory.
     @_size size to be allocated
  */
  inline void alloc(const Size<Dim> &_size)
  {
    DeviceMemoryStorage<Type, Dim>::alloc(_size);
  }

  /**
     Bind the buffer object.
  */
  inline void bind() { CUDA_OPENGL_CHECK(glBindBuffer(target, bufname)); }

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
     Free buffer memory.
  */
  void free();

  inline GLuint getName() const { return bufname; }

  /**
     Unbind the buffer object.
  */
  inline void unbind() { CUDA_OPENGL_CHECK(glBindBuffer(target, 0)); }

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
};

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
connect()
{
  if(this->buffer != 0)
    return;

  CUDA_CHECK(cudaGLRegisterBufferObject(bufname));
  CUDA_CHECK(cudaGLMapBufferObject((void **)&this->buffer, bufname));

  if(this->buffer == 0)
    CUDA_ERROR("map buffer object failed");
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
disconnect()
{
  if(this->buffer == 0)
    return;

  CUDA_CHECK(cudaGLUnmapBufferObject(bufname));
  CUDA_CHECK(cudaGLUnregisterBufferObject(bufname));
  this->buffer = 0;
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
alloc()
{
  this->free();
  this->setPitch(0);
  size_t p = 1;

  for(size_t i = Dim; i--;)
    p *= this->size[i];

  CUDA_OPENGL_CHECK(glGenBuffers(1, &bufname));
  bind();
  CUDA_OPENGL_CHECK(glBufferData(target, p * sizeof(Type), 0, usage));
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
  glBindBuffer(target, 0);
  glDeleteBuffers(1, &bufname);
  bufname = 0;
}

template <class Type, unsigned Dim>
BufferObject<Type, Dim>::
~BufferObject()
{
  this->free();
}

// #include "auto/specdim_opengl_bufferobject.hpp"

}
}


#endif
