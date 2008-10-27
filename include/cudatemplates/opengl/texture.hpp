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

#ifndef CUDA_OPENGL_TEXTURE_H
#define CUDA_OPENGL_TEXTURE_H


#include <GL/gl.h>
#include <GL/glext.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cudatemplates/opengl/type.hpp>
#include <cudatemplates/storage.hpp>


namespace Cuda {
namespace OpenGL {

/**
   Representation of OpenGL texture.
*/
template <class Type, unsigned Dim>
class Texture: public Storage<Type, Dim>
{
  CUDA_STATIC_ASSERT(Dim <= 3);

public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Texture():
    texname(0)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
  */
  inline Texture(const Size<Dim> &_size):
    texname(0),
    Layout<Type, Dim>(_size),
    Storage<Type, Dim>(_size)
  {
    alloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
  */
  inline Texture(const Layout<Type, Dim> &layout):
    texname(0),
    Layout<Type, Dim>(layout),
    Storage<Type, Dim>(layout)
  {
    alloc();
  }

  // #include "auto/copy_opengl_texture.hpp"

  /**
     Allocate GPU memory.
  */
  void alloc();

  /**
     Bind OpenGL buffer object.
  */
  // inline void bind() { glBindBuffer(GL_ARRAY_BUFFER, texname); }

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
  GLuint texname;
};

template <class Type, unsigned Dim>
void Texture<Type, Dim>::
alloc()
{
  for(int i = Dim; i--;)
    if((this->size[i] & (this->size[i] - 1)) != 0)
      CUDA_ERROR("Texture size must be power of two");

  this->free();
  glGenTextures(1, &texname);

  switch(Dim) {
  case 1:
    glBindTexture(GL_TEXTURE_1D, texname);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_LUMINANCE, this->size[0], 0, GL_LUMINANCE, getType<Type>(), 0);
    break;

  case 2:
    glBindTexture(GL_TEXTURE_2D, texname);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, this->size[0], this->size[1], 0, GL_LUMINANCE, getType<Type>(), 0);
    break;

  case 3:
    glBindTexture(GL_TEXTURE_3D, texname);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, this->size[0], this->size[1], this->size[2], 0, GL_LUMINANCE, getType<Type>(), 0);
  }

  // TODO: check for OpenGL error
}

template <class Type, unsigned Dim>
void Texture<Type, Dim>::
free()
{
  if(this->texname == 0)
    return;

  glDeleteTextures(1, &texname);
  texname = 0;
}

// #include "auto/specdim_opengl_texture.hpp"

}
}


#endif
