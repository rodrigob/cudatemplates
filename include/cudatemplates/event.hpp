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

#ifndef CUDA_EVENT_H
#define CUDA_EVENT_H


#include <cudatemplates/error.hpp>


namespace Cuda {

/**
   CUDA runtime event class.
*/
class Event
{
public:
  /**
     Default constructor.
  */
  inline Event() { CUDA_CHECK(cudaEventCreate(&event)); }

  /**
     Destructor.
  */
  inline ~Event() { CUDA_CHECK(cudaEventDestroy(event)); }

  /**
     Computes the elapsed time between events.
     @return elapsed time in milliseconds
  */
  inline float operator-(const Event &e) const
  {
    float time;
    CUDA_CHECK(cudaEventElapsedTime(&time, e.event, event));
    return time;
  }

  /**
     Records an event.
  */
  inline void record(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(event, stream)); }

  /**
     Query if an event has been recorded.
  */
  inline bool query() const
  {
    cudaError_t ret = cudaEventQuery(event);

    if(ret == cudaSuccess)
      return true;
    else if(ret == cudaErrorNotReady)
      return false;

    CUDA_CHECK(ret);
    return false;  // suppress compiler warning
  }

  /**
     Wait for an event to be recorded.
  */
  inline void synchronize() { CUDA_CHECK(cudaEventSynchronize(event)); }

private:
  cudaEvent_t event;
};

}  // namespace Cuda


#endif
