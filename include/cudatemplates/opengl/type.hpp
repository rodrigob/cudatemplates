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


#include <string.h>

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
   @return format parameter
*/
template <class T> inline  GLenum getFormat() { CUDA_OPENGL_ERROR("unsupported texture format"); }

#define CUDA_OPENGL_FORMAT(a, b) template <> inline GLenum getFormat<a>() { return b; }
#define CUDA_OPENGL_FORMAT_CHECKED(a, b, support, alternative) template <> inline GLenum getFormat<a>() { if(support)return b; }

CUDA_OPENGL_FORMAT(unsigned char , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct uchar1 , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct uchar2 , GL_LUMINANCE_ALPHA)
#ifndef GL_RGB_INTEGER
#define GL_RGB_INTEGER 0x8D98
#endif
CUDA_OPENGL_FORMAT(struct uchar3 , GL_RGB_INTEGER)
#ifndef GL_RGBA_INTEGER
#define GL_RGBA_INTEGER 0x8D99
#endif
CUDA_OPENGL_FORMAT(struct uchar4 , GL_RGBA_INTEGER)

CUDA_OPENGL_FORMAT(char          , GL_LUMINANCE_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct char1  , GL_LUMINANCE_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct char2  , GL_LUMINANCE_ALPHA_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct char3  , GL_RGB_INTEGER)
CUDA_OPENGL_FORMAT(struct char4  , GL_RGBA_INTEGER)

CUDA_OPENGL_FORMAT(unsigned short, GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct ushort1, GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct ushort2, GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct ushort3, GL_RGB_INTEGER)
CUDA_OPENGL_FORMAT(struct ushort4, GL_RGBA_INTEGER)

CUDA_OPENGL_FORMAT(short         , GL_LUMINANCE_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct short1 , GL_LUMINANCE_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct short2 , GL_LUMINANCE_ALPHA_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct short3 , GL_RGB_INTEGER)
CUDA_OPENGL_FORMAT(struct short4 , GL_RGBA_INTEGER)

CUDA_OPENGL_FORMAT(unsigned int  , GL_LUMINANCE_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct uint1  , GL_LUMINANCE_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct uint2  , GL_LUMINANCE_ALPHA_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct uint3  , GL_RGB_INTEGER)
CUDA_OPENGL_FORMAT(struct uint4  , GL_RGBA_INTEGER)

CUDA_OPENGL_FORMAT(int           , GL_LUMINANCE_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct int1   , GL_LUMINANCE_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct int2   , GL_LUMINANCE_ALPHA_INTEGER_EXT)
CUDA_OPENGL_FORMAT(struct int3   , GL_RGB_INTEGER)
CUDA_OPENGL_FORMAT(struct int4   , GL_RGBA_INTEGER)

CUDA_OPENGL_FORMAT(float         , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct float1 , GL_LUMINANCE)
CUDA_OPENGL_FORMAT(struct float2 , GL_LUMINANCE_ALPHA)
CUDA_OPENGL_FORMAT(struct float3 , GL_RGB)
CUDA_OPENGL_FORMAT(struct float4 , GL_RGBA)

#undef CUDA_OPENGL_FORMAT
#undef CUDA_OPENGL_FORMAT_CHECKED

//------------------------------------------------------------------------------

/**
   Provide "internalFormat" parameter for OpenGL texture calls.
   @return internalFormat parameter
*/
template <class T> inline GLint getInternalFormat() { CUDA_OPENGL_ERROR("unsupported texture format"); }

#define CUDA_OPENGL_INTERNAL_FORMAT(a, b) template <> inline GLint getInternalFormat<a>() { return b; }

CUDA_OPENGL_INTERNAL_FORMAT(unsigned char , GL_LUMINANCE8)
CUDA_OPENGL_INTERNAL_FORMAT(struct uchar1 , GL_LUMINANCE8)
CUDA_OPENGL_INTERNAL_FORMAT(struct uchar2 , GL_LUMINANCE8_ALPHA8)
#ifndef GL_RGB8UI
#define GL_RGB8UI 0x8D7D
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct uchar3 , GL_RGB8UI)
#ifndef GL_RGBA8UI
#define GL_RGBA8UI 0x8D7C
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct uchar4 , GL_RGBA8UI)

CUDA_OPENGL_INTERNAL_FORMAT(char          , GL_LUMINANCE8I_EXT)
CUDA_OPENGL_INTERNAL_FORMAT(struct char1  , GL_LUMINANCE8I_EXT)
CUDA_OPENGL_INTERNAL_FORMAT(struct char2  , GL_LUMINANCE_ALPHA8I_EXT)
#ifndef GL_RGB8I
#define GL_RGB8I 0x8D8F
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct char3  , GL_RGB8I)
#ifndef GL_RGBA8I
#define GL_RGBA8I 0x8D8E
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct char4  , GL_RGBA8I)

CUDA_OPENGL_INTERNAL_FORMAT(unsigned short, GL_LUMINANCE16)
CUDA_OPENGL_INTERNAL_FORMAT(struct ushort1, GL_LUMINANCE16)
CUDA_OPENGL_INTERNAL_FORMAT(struct ushort2, GL_LUMINANCE16_ALPHA16)
CUDA_OPENGL_INTERNAL_FORMAT(struct ushort3, GL_RGB16)
CUDA_OPENGL_INTERNAL_FORMAT(struct ushort4, GL_RGBA16)

CUDA_OPENGL_INTERNAL_FORMAT(short         , GL_LUMINANCE16I_EXT)
CUDA_OPENGL_INTERNAL_FORMAT(struct short1 , GL_LUMINANCE16I_EXT)
CUDA_OPENGL_INTERNAL_FORMAT(struct short2 , GL_LUMINANCE_ALPHA16I_EXT)
#ifndef GL_RGB16I
#define GL_RGB16I 0x8D89
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct short3 , GL_RGB16I)
#ifndef GL_RGBA16I
#define GL_RGBA16I 0x8D88
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct short4 , GL_RGBA16I)

CUDA_OPENGL_INTERNAL_FORMAT(unsigned int  , GL_LUMINANCE32UI_EXT)
CUDA_OPENGL_INTERNAL_FORMAT(struct uint1  , GL_LUMINANCE32UI_EXT)
CUDA_OPENGL_INTERNAL_FORMAT(struct uint2  , GL_LUMINANCE_ALPHA32UI_EXT)
#ifndef GL_RGB32UI
#define GL_RGB32UI 0x8D71
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct uint3  , GL_RGB32UI)
#ifndef GL_RGBS32UI
#define GL_RGBA32UI 0x8D70
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct uint4  , GL_RGBA32UI)

CUDA_OPENGL_INTERNAL_FORMAT(int           , GL_LUMINANCE32I_EXT)
CUDA_OPENGL_INTERNAL_FORMAT(struct int1   , GL_LUMINANCE32I_EXT)
CUDA_OPENGL_INTERNAL_FORMAT(struct int2   , GL_LUMINANCE_ALPHA32I_EXT)
#ifndef GL_RGB32I
#define GL_RGB32I 0x8D83
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct int3   , GL_RGB32I)
#ifndef GL_RGBA32I
#define GL_RGBA32I 0x8D82
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct int4   , GL_RGBA32I)

CUDA_OPENGL_INTERNAL_FORMAT(float         , GL_LUMINANCE32F_ARB)
CUDA_OPENGL_INTERNAL_FORMAT(struct float1 , GL_LUMINANCE32F_ARB)
CUDA_OPENGL_INTERNAL_FORMAT(struct float2 , GL_LUMINANCE_ALPHA32F_ARB)
#ifndef GL_RGB32F
#define GL_RGB32F 0x8815
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct float3 , GL_RGB32F)
#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif
CUDA_OPENGL_INTERNAL_FORMAT(struct float4 , GL_RGBA32F)

#undef CUDA_OPENGL_INTERNAL_FORMAT


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


//------------------------------------------------------------------------------
static bool ext_texture_integer_supported()
{
  const GLubyte *str = glGetString(GL_EXTENSIONS);
  return strstr((const char *)str, "GL_EXT_texture_integer") != 0;
}

static bool arb_texture_float_supported()
{
  const GLubyte *str = glGetString(GL_EXTENSIONS);
  return strstr((const char *)str, "GL_ARB_texture_float") != 0;
}

//------------------------------------------------------------------------------

/**
   Method to check if texture format is supported
   @return true if supported
*/

template <class T> inline bool formatSupported() { CUDA_OPENGL_ERROR("unsupported texture type"); }

#define CUDA_OPENGL_FORMAT_SUPPORTED(a, b) template <> inline bool formatSupported<a>() { return b; }

CUDA_OPENGL_FORMAT_SUPPORTED(unsigned char , true)
CUDA_OPENGL_FORMAT_SUPPORTED(struct uchar1 , true)
CUDA_OPENGL_FORMAT_SUPPORTED(struct uchar2 , true)
CUDA_OPENGL_FORMAT_SUPPORTED(struct uchar3 , true)
CUDA_OPENGL_FORMAT_SUPPORTED(struct uchar4 , true)

CUDA_OPENGL_FORMAT_SUPPORTED(char          , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct char1  , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct char2  , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct char3  , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct char4  , ext_texture_integer_supported())

CUDA_OPENGL_FORMAT_SUPPORTED(unsigned short, true)
CUDA_OPENGL_FORMAT_SUPPORTED(struct ushort1, true)
CUDA_OPENGL_FORMAT_SUPPORTED(struct ushort2, true)
CUDA_OPENGL_FORMAT_SUPPORTED(struct ushort3, true)
CUDA_OPENGL_FORMAT_SUPPORTED(struct ushort4, true)

CUDA_OPENGL_FORMAT_SUPPORTED(short         , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct short1 , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct short2 , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct short3 , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct short4 , ext_texture_integer_supported())

CUDA_OPENGL_FORMAT_SUPPORTED(unsigned int  , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct uint1  , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct uint2  , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct uint3  , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct uint4  , ext_texture_integer_supported())

CUDA_OPENGL_FORMAT_SUPPORTED(int           , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct int1   , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct int2   , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct int3   , ext_texture_integer_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct int4   , ext_texture_integer_supported())

CUDA_OPENGL_FORMAT_SUPPORTED(float         , arb_texture_float_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct float1 , arb_texture_float_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct float2 , arb_texture_float_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct float3 , arb_texture_float_supported())
CUDA_OPENGL_FORMAT_SUPPORTED(struct float4 , arb_texture_float_supported())

#undef CUDA_OPENGL_FORMAT_SUPPORTED

}
}


#endif
