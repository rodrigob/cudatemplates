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

  // #include "auto/copy_opengl_bufferobject.hpp"

  /**
     Allocate GPU memory.
  */
  void alloc();

  /**
     Bind OpenGL buffer object.
  */
  inline void bind() { glBindBuffer(GL_ARRAY_BUFFER, bufname); }

  /**
     Free GPU memory.
  */
  void free();

  /**
     Unbind OpenGL buffer object.
  */
  inline void unbind() { glBindBuffer(GL_ARRAY_BUFFER, 0); }


private:
  /**
     Buffer object name.
  */
  GLuint bufname;
};

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
alloc()
{
  this->free();
  size_t p = 1;

  for(size_t i = Dim; i--;)
    p *= this->size[i];

  /*
    TODO:
    -) check for OpenGL and CUDA errors
    -) make OpenGL buffer type configurable
  */
  glGenBuffers(1, &bufname);
  bind();
  glBufferData(GL_ARRAY_BUFFER, p * sizeof(Type), 0, GL_DYNAMIC_DRAW);

  cudaGLRegisterBufferObject(bufname);
  cudaGLMapBufferObject((void **)&this->buffer, bufname);

  this->setPitch(0);
  assert(this->buffer != 0);
}

template <class Type, unsigned Dim>
void BufferObject<Type, Dim>::
free()
{
  if(this->bufname == 0)
    return;

  cudaGLUnmapBufferObject(bufname);
  cudaGLUnregisterBufferObject(bufname);
  unbind();
  glDeleteBuffers(1, &bufname);
  bufname = 0;
  this->buffer = 0;
}

// #include "auto/specdim_opengl_bufferobject.hpp"

}

}


#endif
