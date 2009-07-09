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

#ifndef CUDA_STREAM_H
#define CUDA_STREAM_H


#include <cudatemplates/error.hpp>


namespace Cuda {

/**
   CUDA stream class.
*/
class Stream
{
public:
  /**
     Default constructor.
  */
  inline Stream() { CUDA_CHECK(cudaStreamCreate(&stream)); }

  /**
     Destructor.
  */
  inline ~Stream() { CUDA_CHECK(cudaStreamDestroy(stream)); }

  /**
     Get stream identifier.
  */
  inline operator cudaStream_t() const { return stream; }

  /**
     Query if a stream has finished.
  */
  inline bool query() const
  {
    cudaError_t ret = cudaStreamQuery(stream);

    if(ret == cudaSuccess)
      return true;
    else if(ret == cudaErrorNotReady)
      return false;

    CUDA_CHECK(ret);
    return false;  // suppress compiler warning
  }

  /**
     Wait for an stream to finish.
  */
  inline void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream)); }

private:
  cudaStream_t stream;
};

}  // namespace Cuda


#endif
