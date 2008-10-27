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


namespace Cuda {
namespace OpenGL {

template <class Type> GLenum getType();

#define CUDA_OPENGL_TYPE(a, b) template <> inline GLenum getType<a>() { return b; }

CUDA_OPENGL_TYPE(unsigned char , GL_UNSIGNED_BYTE)
CUDA_OPENGL_TYPE(         char , GL_BYTE)
CUDA_OPENGL_TYPE(unsigned short, GL_UNSIGNED_SHORT)
CUDA_OPENGL_TYPE(         short, GL_SHORT)
CUDA_OPENGL_TYPE(unsigned int  , GL_UNSIGNED_INT)
CUDA_OPENGL_TYPE(         int  , GL_INT)
CUDA_OPENGL_TYPE(         float, GL_FLOAT)

}
}


#endif
