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
#include <GL/glext.h>

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
    bufname(0)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
  */
  inline BufferObject(const Size<Dim> &_size):
    bufname(0),
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemoryStorage<Type, Dim>(_size)
  {
    alloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
  */
  inline BufferObject(const Layout<Type, Dim> &layout):
    bufname(0),
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemoryStorage<Type, Dim>(layout)
  {
    alloc();
  }

  ~BufferObject();

  // #include "auto/copy_opengl_bufferobject.hpp"

  /**
     Allocate GPU memory.
  */
  void alloc();

  /**
     Bind OpenGL buffer object.
  */
  // inline void bind() { glBindBuffer(GL_ARRAY_BUFFER, bufname); }

  /**
     Register and map buffer object.
     If you called disconnect(), this must be called before using the buffer
     memory in a CUDA kernel.
  */
  void connect();

  /**
     Copy OpenGL texture to buffer object.
     @param texname texture name
  */
  void copyFromTexture(GLuint texname);

  /**
     Copy buffer object to OpenGL texture.
     @param texname texture name
  */
  void copyToTexture(GLuint texname);

  /**
     Unmap and unregister buffer object.
     This must be called before accessing the buffer memory in OpenGL, e.g.,
     copying the buffer data from or to a texture.
  */
  void disconnect();

  /**
     Free GPU memory.
  */
  void free();

  /**
     Unbind OpenGL buffer object.
  */
  // inline void unbind() { glBindBuffer(GL_ARRAY_BUFFER, 0); }


private:
  /**
     Buffer object name.
  */
  GLuint bufname;
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
  CUDA_OPENGL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, bufname));
  CUDA_OPENGL_CHECK(glBufferData(GL_ARRAY_BUFFER, p * sizeof(Type), 0, GL_DYNAMIC_DRAW));
  CUDA_OPENGL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
  connect();
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
copyFromTexture(GLuint texname)
{
  // TODO: currently hard-coded for one channel and two dimensions
  disconnect();
  CUDA_OPENGL_CHECK(glBindBuffer(GL_PIXEL_PACK_BUFFER, bufname));
  CUDA_OPENGL_CHECK(glReadPixels(0, 0, this->size[0], this->size[1], GL_LUMINANCE, getType<Type>(), NULL));
  CUDA_OPENGL_CHECK(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
  connect();
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
copyToTexture(GLuint texname)
{
  // TODO: currently hard-coded for one channel
  GLenum target;

  switch(Dim) {
  case 1: target = GL_TEXTURE_1D; break;
  case 2: target = GL_TEXTURE_2D; break;
  case 3: target = GL_TEXTURE_3D;
  }

  disconnect();
  CUDA_OPENGL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufname));
  CUDA_OPENGL_CHECK(glBindTexture(target, texname));

  switch(Dim) {
  case 1:
    CUDA_OPENGL_CHECK(glTexSubImage1D(target, 0,
				      0, this->size[0],
				      GL_LUMINANCE, getType<Type>(), NULL));
    break;

  case 2:
    CUDA_OPENGL_CHECK(glTexSubImage2D(target, 0,
				      0, 0, this->size[0], this->size[1],
				      GL_LUMINANCE, getType<Type>(), NULL));
    break;

  case 3:
    CUDA_OPENGL_CHECK(glTexSubImage3D(target, 0,
				      0, 0, 0, this->size[0], this->size[1], this->size[2],
				      GL_LUMINANCE, getType<Type>(), NULL));
  }

  CUDA_OPENGL_CHECK(glBindTexture(target, 0));
  CUDA_OPENGL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
  connect();
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
free()
{
  if(this->bufname == 0)
    return;

  disconnect();
  glBindBuffer(GL_ARRAY_BUFFER, 0);
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
