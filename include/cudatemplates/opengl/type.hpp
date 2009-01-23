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

#ifndef CUDA_OPENGL_TYPE_H
#define CUDA_OPENGL_TYPE_H


#include <GL/gl.h>

#include <vector_types.h>

#include <cudatemplates/opengl/error.hpp>


namespace Cuda {
namespace OpenGL {

template <class T> GLint getInternalFormat();

//------------------------------------------------------------------------------

/**
   Provide "format" parameter for OpenGL texture calls.
   In some situations "GL_BGR" is more efficient than "GL_RGB".
   However, "GL_BGR" is not yet supported.
   Moreover, in the current implementation, "format" and "internalFormat" are
   identical (see getInternalFormat), but there is no internal format "GL_BGR".
   @return format parameter
*/
template <class T> inline  GLenum getFormat() { CUDA_OPENGL_ERROR("unsupported texture format"); }

#define CUDA_OPENGL_FORMAT(a, b) template <> inline GLenum getFormat<a>() { return b; }

CUDA_OPENGL_FORMAT(unsigned char , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct uchar1 , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct uchar2 , GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct uchar3 , GL_RGB)
CUDA_OPENGL_FORMAT(struct uchar4 , GL_RGBA)

CUDA_OPENGL_FORMAT(char          , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct char1  , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct char2  , GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct char3  , GL_RGB)
CUDA_OPENGL_FORMAT(struct char4  , GL_RGBA)

CUDA_OPENGL_FORMAT(unsigned short, GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct ushort1, GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct ushort2, GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct ushort3, GL_RGB)
CUDA_OPENGL_FORMAT(struct ushort4, GL_RGBA)

CUDA_OPENGL_FORMAT(short         , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct short1 , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct short2 , GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct short3 , GL_RGB)
CUDA_OPENGL_FORMAT(struct short4 , GL_RGBA)

CUDA_OPENGL_FORMAT(unsigned int  , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct uint1  , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct uint2  , GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct uint3  , GL_RGB)
CUDA_OPENGL_FORMAT(struct uint4  , GL_RGBA)

CUDA_OPENGL_FORMAT(int           , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct int1   , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct int2   , GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct int3   , GL_RGB)
CUDA_OPENGL_FORMAT(struct int4   , GL_RGBA)

CUDA_OPENGL_FORMAT(float         , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct float1 , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct float2 , GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct float3 , GL_RGB)
CUDA_OPENGL_FORMAT(struct float4 , GL_RGBA)

#undef CUDA_OPENGL_FORMAT

//------------------------------------------------------------------------------

/**
   Provide "internalFormat" parameter for OpenGL texture calls.
   In the current implementation, this is identical with the "format" parameter.
   @return internalFormat parameter
*/
template <class T> inline GLint getInternalFormat() { return (GLint)getFormat<T>(); }

//------------------------------------------------------------------------------

/**
   Provide "type" parameter for OpenGL texture calls.
   @return type parameter
*/
template <class T> inline GLenum getType() { CUDA_OPENGL_ERROR("unsupported texture type"); }

#define CUDA_OPENGL_TYPE(a, b) template <> inline GLenum getType<a>() { return b; }

CUDA_OPENGL_TYPE(unsigned char , GL_UNSIGNED_BYTE)
CUDA_OPENGL_TYPE(struct uchar1 , GL_UNSIGNED_BYTE)
CUDA_OPENGL_TYPE(struct uchar2 , GL_UNSIGNED_BYTE)
CUDA_OPENGL_TYPE(struct uchar3 , GL_UNSIGNED_BYTE)
CUDA_OPENGL_TYPE(struct uchar4 , GL_UNSIGNED_BYTE)

CUDA_OPENGL_TYPE(char          , GL_BYTE)
CUDA_OPENGL_TYPE(struct char1  , GL_BYTE)
CUDA_OPENGL_TYPE(struct char2  , GL_BYTE)
CUDA_OPENGL_TYPE(struct char3  , GL_BYTE)
CUDA_OPENGL_TYPE(struct char4  , GL_BYTE)

CUDA_OPENGL_TYPE(unsigned short, GL_UNSIGNED_SHORT)
CUDA_OPENGL_TYPE(struct ushort1, GL_UNSIGNED_SHORT)
CUDA_OPENGL_TYPE(struct ushort2, GL_UNSIGNED_SHORT)
CUDA_OPENGL_TYPE(struct ushort3, GL_UNSIGNED_SHORT)
CUDA_OPENGL_TYPE(struct ushort4, GL_UNSIGNED_SHORT)

CUDA_OPENGL_TYPE(short         , GL_SHORT)
CUDA_OPENGL_TYPE(struct short1 , GL_SHORT)
CUDA_OPENGL_TYPE(struct short2 , GL_SHORT)
CUDA_OPENGL_TYPE(struct short3 , GL_SHORT)
CUDA_OPENGL_TYPE(struct short4 , GL_SHORT)

CUDA_OPENGL_TYPE(unsigned int  , GL_UNSIGNED_INT)
CUDA_OPENGL_TYPE(struct uint1  , GL_UNSIGNED_INT)
CUDA_OPENGL_TYPE(struct uint2  , GL_UNSIGNED_INT)
CUDA_OPENGL_TYPE(struct uint3  , GL_UNSIGNED_INT)
CUDA_OPENGL_TYPE(struct uint4  , GL_UNSIGNED_INT)

CUDA_OPENGL_TYPE(int           , GL_INT)
CUDA_OPENGL_TYPE(struct int1   , GL_INT)
CUDA_OPENGL_TYPE(struct int2   , GL_INT)
CUDA_OPENGL_TYPE(struct int3   , GL_INT)
CUDA_OPENGL_TYPE(struct int4   , GL_INT)

CUDA_OPENGL_TYPE(float         , GL_FLOAT)
CUDA_OPENGL_TYPE(struct float1 , GL_FLOAT)
CUDA_OPENGL_TYPE(struct float2 , GL_FLOAT)
CUDA_OPENGL_TYPE(struct float3 , GL_FLOAT)
CUDA_OPENGL_TYPE(struct float4 , GL_FLOAT)

#undef CUDA_OPENGL_TYPE

}
}


#endif
