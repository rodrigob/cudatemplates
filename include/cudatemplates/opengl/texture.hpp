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


#include <string.h>

#include <GL/gl.h>

#ifndef _WIN32
#include <GL/glext.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cudatemplates/opengl/error.hpp>
#include <cudatemplates/opengl/type.hpp>
#include <cudatemplates/storage.hpp>


namespace Cuda {
namespace OpenGL {

/**
   Representation of OpenGL texture.
*/
template <class Type, unsigned Dim>
class Texture:
    virtual public Layout<Type, Dim>,
    public Storage<Type, Dim>
{
  CUDA_STATIC_ASSERT(Dim <= 3);

public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Texture()
  {
    init();
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
  */
  inline Texture(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Storage<Type, Dim>(_size)
  {
    init();
    allocInternal();
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
    allocInternal();
  }

  /**
     Destructor.
  */
  ~Texture()
  {
    this->free();
  }

  /**
     Bind OpenGL texture object.
  */
  inline void bind()
  {
    CUDA_OPENGL_CHECK(glBindTexture(target(), texname));
  }

  inline GLuint getName() const { return texname; }

  /**
     Determine if memory is currently allocated.
  */
  bool isAllocated() const;

  /**
     Get OpenGL texture target.
  */
  static inline GLenum target()
  {
    switch(Dim) {
    case 1: return GL_TEXTURE_1D;
    case 2: return GL_TEXTURE_2D;
    case 3: return GL_TEXTURE_3D;
    }

    return 0;  // satisfy compiler
  }

  void glTexSubImage(const GLvoid *pixels);

  /**
     Unbind OpenGL buffer object.
  */
  static inline void unbind()
  {
    CUDA_OPENGL_CHECK(glBindTexture(target(), 0));
  }

private:
  /**
     Buffer object name.
  */
  GLuint texname;

  /**
     Allocate texture memory.
  */
  void allocInternal();

  /**
     Free texture memory.
  */
  void freeInternal();

  /**
     Initialize texture name.
  */
  void init();
};

template <class Type, unsigned Dim>
void Texture<Type, Dim>::
allocInternal()
{
  // check for OpenGL extensions:
  static bool init_extensions = false;
  static bool has_npot_extension;

  if(!init_extensions) {
    const GLubyte *str = glGetString(GL_EXTENSIONS);
    has_npot_extension = (strstr((const char *)str, "GL_ARB_texture_non_power_of_two") != 0);

    //check if format is supported
    if(!formatSupported<Type>())
      CUDA_ERROR("Texture format not supported");

    init_extensions = true;
  }

  // if not available, check for power-of-two texture image dimensions:
  if(!has_npot_extension)
    for(int i = Dim; i--;)
      if((this->size[i] & (this->size[i] - 1)) != 0)
	CUDA_ERROR("Texture size must be power of two");

  this->free();
  CUDA_OPENGL_CHECK(glGenTextures(1, &texname));
  CUDA_OPENGL_CHECK(glBindTexture(target(), texname));

  GLint internalFormat = getInternalFormat<Type>();
  GLenum format = getFormat<Type>();
  GLenum type = getType<Type>();

  switch(Dim) {
  case 1:
    CUDA_OPENGL_CHECK(glTexImage1D(target(), 0, internalFormat, this->size[0], 0, format, type, 0));
    break;

  case 2:
    CUDA_OPENGL_CHECK(glTexImage2D(target(), 0, internalFormat, this->size[0], this->size[1], 0, format, type, 0));
    break;

  case 3:
    CUDA_OPENGL_CHECK(glTexImage3D(target(), 0, internalFormat, this->size[0], this->size[1], this->size[2], 0, format, type, 0));
  }

  // A minimal set of texture parameters is required, otherwise the texture
  // won't be visible at all (feel free to override them later as needed):
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

template <class Type, unsigned Dim>
void Texture<Type, Dim>::
glTexSubImage(const GLvoid *pixels)
{
  bind();

  GLenum format = getFormat<Type>();
  GLenum type = getType<Type>();

  switch(Dim) {
  case 1:
    CUDA_OPENGL_CHECK(glTexSubImage1D(target(), 0,
				      0, this->size[0],
				      format, type, pixels));
    break;

  case 2:
    CUDA_OPENGL_CHECK(glTexSubImage2D(target(), 0,
				      0, 0, this->size[0], this->size[1],
				      format, type, pixels));
    break;

  case 3:
    CUDA_OPENGL_CHECK(glTexSubImage3D(target(), 0,
				      0, 0, 0, this->size[0], this->size[1], this->size[2],
				      format, type, pixels));
  }

  unbind();
}

template <class Type, unsigned Dim>
void Texture<Type, Dim>::
freeInternal()
{
  CUDA_OPENGL_CHECK(glDeleteTextures(1, &texname));
}

template <class Type, unsigned Dim>
void Texture<Type, Dim>::
init()
{
  texname = 0;
}

template <class Type, unsigned Dim>
bool Texture<Type, Dim>::
isAllocated() const
{
  return texname != 0;
}

}  // namespace OpenGL
}  // namespace Cuda


#include "../auto/specdim_texture.hpp"


#endif
