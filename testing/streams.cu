/* 
  Cuda Templates.

  Copyright (C) 2009 Institute for Computer Graphics and Vision,
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

#include <iostream>

#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/event.hpp>
#include <cudatemplates/stream.hpp>

using namespace std;


const int NUM_STREAMS = 4;
const int NUM_PASSES = 4;


__global__ void
kernel(Cuda::DeviceMemoryLinear1D<float>::KernelData dst,
       Cuda::DeviceMemoryLinear1D<float>::KernelData src)
{
  // do some stupid copying to waste time
  for(int i = 0; i < dst.size[0]; ++i)
    dst.data[i] = src.data[i];
}

inline int
event_id(int stream, int pass)
{
  return stream * (NUM_PASSES + 1) + pass;
}

int
main()
{
  try {
    Cuda::Size<1> size(1 << 20);
    Cuda::DeviceMemoryLinear1D<float> src[NUM_STREAMS], dst[NUM_STREAMS];

    for(int i = NUM_STREAMS; i--;) {
      src[i].alloc(size);
      dst[i].alloc(size);
    }

    Cuda::Stream stream[NUM_STREAMS];
    Cuda::Event event[NUM_STREAMS * (NUM_PASSES + 1)];
    dim3 gridDim(1), blockDim(1);

    for(int i = 0; i < NUM_STREAMS; ++i) {
      Cuda::Stream &s = stream[i];
      event[event_id(i, 0)].record(s);

      for(int j = 0; j < NUM_PASSES; ++j) {
	kernel<<<gridDim, blockDim, 0, s>>>(dst[i], src[i]);
	event[event_id(i, j + 1)].record(s);
      }
    }

    cudaThreadSynchronize();

    for(int i = 0; i < NUM_STREAMS; ++i) {
      for(int j = 0; j < NUM_PASSES; ++j) {
	float t = event[event_id(i, j + 1)] - event[event_id(0, 0)];
	cout << "stream " << i << " / event " << (j + 1) << " recorded at " << t << "ms\n";
      }
    }
  }
  catch(const exception &e) {
    cerr << e.what() << endl;
    return 1;
  }

  return 0;
}
