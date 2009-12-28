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

#ifndef CUDA_GRAPHICS_OPENGL_TEXTURE_H
#define CUDA_GRAPHICS_OPENGL_TEXTURE_H


#include <assert.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

// #include <cudatemplates/devicememory.hpp>
#include <cudatemplates/error.hpp>
#include <cudatemplates/graphics/opengl/image.hpp>
// #include <cudatemplates/layout.hpp>
#include <cudatemplates/opengl/error.hpp>
#include <cudatemplates/opengl/type.hpp>


namespace Cuda {
namespace Graphics {
namespace OpenGL {

template <class Type, unsigned Dim>
class Texture:
    virtual public Layout<Type, Dim>,
    public Image<Type, Dim>
{
public:
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

#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
     @param t target to which the texture object is bound
     @param u usage pattern of the data store
  */
  inline Texture():
    Layout<Type, Dim>(),
    Image<Type, Dim>(0)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
     @param t target to which the texture object is bound
     @param u usage pattern of the data store
  */
  inline Texture(const Size<Dim> &_size, unsigned int f = 0):
    Layout<Type, Dim>(_size),
    Image<Type, Dim>(_size, f)
  {
    realloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
     @param t target to which the texture object is bound
     @param u usage pattern of the data store
  */
  inline Texture(const Layout<Type, Dim> &layout, unsigned int f = 0):
    Layout<Type, Dim>(layout),
    Image<Type, Dim>(layout, f)
  {
    realloc();
  }

  ~Texture();

  // #include "auto/copy_opengl_textureobject.hpp"

  /**
     Bind the texture object to the given target.
     @param t target to which the texture object should be bound
  */
  /*
  inline void bind(GLenum t)
  {
    CUDA_OPENGL_CHECK(glBindTexture(t, this->name));
  }
  */

  /**
     Free texture memory.
  */
  void free();

  /**
     Allocate texture memory.
  */
  void realloc();

  /**
     Allocate texture memory.
     @_size size to be allocated
  */
  /*
  inline void realloc(const Size<Dim> &_size)
  {
    DeviceMemoryStorage<Type, Dim>::realloc(_size);
  }
  */

  /**
     Unbind the texture object from the given target.
     @param t target from which the texture object should be unbound
  */
  /*
  inline static void unbind(GLenum t) {
    CUDA_OPENGL_CHECK(glBindTexture(t, 0));
  }
  */

#ifdef CUDA_GRAPHICS_COMPATIBILITY
  inline void bind() { Graphics::Resource::bind(); }
#endif

private:
  /**
     Bind the texture.
  */
  void bindObjectInternal()
  {
    CUDA_OPENGL_CHECK(glBindTexture(target(), this->name));
  }

  /**
     Register OpenGL texture for use with CUDA.
  */
  void registerObject()
  {
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&this->resource, this->name, target(), this->flags));

    if(this->resource == 0)
      CUDA_ERROR("register texture failed");
  }

  /**
     Unbind the texture.
  */
  inline void unbindObjectInternal()
  {
    CUDA_OPENGL_CHECK(glBindTexture(target(), 0));
  }
};

template <class Type, unsigned Dim>
void Texture<Type, Dim>::
realloc()
{
  using namespace Cuda::OpenGL;

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
  CUDA_OPENGL_CHECK(glGenTextures(1, &(this->name)));

  if(this->name == 0)
    CUDA_ERROR("generate texture failed");

  // do an explicit bind/unbind since the state "INACTIVE" can only be entered
  // after data has been allocated with glTextureData:
  bindObjectInternal();

  // see "http://forums.nvidia.com/index.php?showtopic=151892" for format issues
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

  unbindObjectInternal();

  // prepare object for use with CUDA:
  this->setState(Graphics::Resource::STATE_CUDA_MAPPED);
}

template <class Type, unsigned Dim>
void Texture<Type, Dim>::
free()
{
  if(this->name == 0)
    return;

  this->setState(Graphics::Resource::STATE_UNUSED);
  CUDA_OPENGL_CHECK(glDeleteTextures(1, &(this->name)));
  this->name = 0;
}

template <class Type, unsigned Dim>
Texture<Type, Dim>::
~Texture()
{
  this->free();
}

}  // namespace OpenGL
}  // namespace Graphics
}  // namespace Cuda


#endif
