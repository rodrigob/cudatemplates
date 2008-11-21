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


#include <time.h>

#include <cstdlib>
#include <iostream>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/hostmemoryheap.hpp>

using namespace std;


const size_t SIZE[] = { 16, 32, 64 };


void
check(float *buf, bool init)
{
  static int seed;
  size_t size = SIZE[0] * SIZE[1] * SIZE[2];

  if(init) {
    seed = time(0);
    srand(seed);

    while(size--)
      *(buf++) = rand();
  }
  else {
    srand(seed);

    while(size--)
      assert(*(buf++) == rand());
  }
}

void
demo_plain()
{
  // allocate host memory:
  float *mem_host1 = (float *)malloc(sizeof(float) * SIZE[0] * SIZE[1] * SIZE[2]);
  float *mem_host2 = (float *)malloc(sizeof(float) * SIZE[0] * SIZE[1] * SIZE[2]);

  if((mem_host1 == 0) || (mem_host2 == 0)) {
    cerr << "out of memory\n";
    exit(1);
  }
  
  // init host memory:
  check(mem_host1, true);

  // allocate device memory:
  cudaExtent extent;
  extent.width = SIZE[0];
  extent.height = SIZE[1];
  extent.depth = SIZE[2];
  cudaPitchedPtr mem_device;
  CUDA_CHECK(cudaMalloc3D(&mem_device, extent));

  // copy from host memory to device memory:
  cudaMemcpy3DParms p = { 0 };
  p.srcPtr.ptr = mem_host1;
  p.srcPtr.pitch = SIZE[0] * sizeof(float);
  p.srcPtr.xsize = SIZE[0];
  p.srcPtr.ysize = SIZE[1];
  p.dstPtr.ptr = mem_device.ptr;
  p.dstPtr.pitch = mem_device.pitch;
  p.dstPtr.xsize = SIZE[0];
  p.dstPtr.ysize = SIZE[1];
  p.extent.width = SIZE[0] * sizeof(float);
  p.extent.height = SIZE[1];
  p.extent.depth = SIZE[2];
  p.kind = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpy3D(&p));

  // copy from device memory to host memory:
  p.srcPtr.ptr = mem_device.ptr;
  p.srcPtr.pitch = mem_device.pitch;
  p.dstPtr.ptr = mem_host2;
  p.dstPtr.pitch = SIZE[0] * sizeof(float);
  p.kind = cudaMemcpyDeviceToHost;
  CUDA_CHECK(cudaMemcpy3D(&p));

  // verification:
  check(mem_host2, false);

  // free memory:
  CUDA_CHECK(cudaFree(mem_device.ptr));
  free(mem_host2);
  free(mem_host1);
}

void
demo_cudatemplates()
{
  try {
    // allocate host memory:
    Cuda::HostMemoryHeap3D<float> mem_host1(SIZE[0], SIZE[1], SIZE[2]);
    Cuda::HostMemoryHeap3D<float> mem_host2(SIZE[0], SIZE[1], SIZE[2]);

    // init host memory:
    check(mem_host1.getBuffer(), true);

    // allocate device memory:
    Cuda::DeviceMemoryPitched3D<float> mem_device(SIZE[0], SIZE[1], SIZE[2]);

    // copy from host memory to device memory:
    copy(mem_device, mem_host1);

    // copy from device memory to host memory:
    copy(mem_host2, mem_device);

    // verification:
    check(mem_host2.getBuffer(), false);
  }
  catch(const exception &e) {
    cerr << e.what();
  }
}

int
main()
{
  demo_plain();
  demo_cudatemplates();
  return 0;
}
