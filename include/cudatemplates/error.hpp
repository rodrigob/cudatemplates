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

#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H


#ifdef __CUDACC__

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call) { cudaError_t err = call; if(err != cudaSuccess) abort(); }
#define CUDA_ERROR(msg) { std::cerr << msg << std::endl; abort(); }

#else  // __CUDACC__

#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <driver_types.h>

#ifndef __GNUC__
#define __PRETTY_FUNCTION__ "(unknown function)"
#endif

#define CUDA_CHECK(call) { cudaError_t err = call; if(err != cudaSuccess) throw Cuda::Error(__FILE__, __LINE__, __PRETTY_FUNCTION__, (int)err, 0); }
#define CUDA_ERROR(msg) { std::ostringstream s; s << msg; throw Cuda::Error(__FILE__, __LINE__, __PRETTY_FUNCTION__, 0, s.str().c_str()); }


namespace Cuda {

class Error: public std::exception
{
public:
  Error(const char *file, int line, const char *function, int code, const char *comment)
  {
    std::ostringstream s;
    s << "CUDA error in " << file << ':' << line << std::endl << function << std::endl;
    
    if(code != 0)
      s << '#' << code << ": " << cudaGetErrorString((cudaError_t)code) << std::endl;

    if(comment != 0)
      s << comment << std::endl;
    
    message = s.str();
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

};

#endif  // __CUDACC__


#endif
