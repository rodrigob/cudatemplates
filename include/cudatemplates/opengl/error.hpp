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

#ifndef CUDA_OPENGL_ERROR_H
#define CUDA_OPENGL_ERROR_H


#include <GL/gl.h>
#include <GL/glu.h>


#if !defined(__GNUC__) && !defined(__PRETTY_FUNCTION__) 
#define __PRETTY_FUNCTION__ "(unknown function)"
#endif


#if defined(__CUDACC__) || defined(NVCC) || defined(CUDA_SIMPLE_ERROR)

#include <stdio.h>
#include <stdlib.h>

#define CUDA_OPENGL_CHECK(call) {	\
    call;				\
    GLenum err = glGetError();		\
    if(err != GL_NO_ERROR)		\
      abort();				\
}

#define CUDA_OPENGL_ERROR(msg) CUDA_ERROR(msg)

#else  // defined(__CUDACC__) || defined(NVCC)

#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <driver_types.h>

#define CUDA_OPENGL_CHECK(call) {					\
    call;								\
    GLenum err = glGetError();						\
    if(err != GL_NO_ERROR)						\
      throw ::Cuda::OpenGL::Error(__FILE__, __LINE__, __PRETTY_FUNCTION__, (int)err, 0); \
}

#define CUDA_OPENGL_ERROR(msg) { std::ostringstream s; s << msg; throw ::Cuda::Error(__FILE__, __LINE__, __PRETTY_FUNCTION__, 0, s.str().c_str()); }


namespace Cuda {
namespace OpenGL {

class Error: public std::exception
{
public:
  Error(const char *file, int line, const char *function, GLenum code, const char *comment) throw()
  {
    std::ostringstream s;
    s << "OpenGL error in " << file << ':' << line << std::endl << function << std::endl;
    
    if(code != 0)
      s << '#' << code << ": " << (const char *)gluErrorString(code) << std::endl;

    if(comment != 0)
      s << comment << std::endl;
    
    message = s.str();
	std::cout << s.str();
  }

  ~Error() throw()
  {
  }

  const char *what() const throw()
  {
    return message.c_str();
  }

private:
  std::string message;
};

}  // namespace OpenGL
}  // namespace Cuda

#endif  // defined(__CUDACC__) || defined(NVCC)


#endif
