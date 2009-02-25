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

#include <typeinfo>

#include <stdio.h>

#include <cudatemplates/array.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/copy_constant.hpp>
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/event.hpp>


const int BLOCK_SIZE = 16;
const int SIZE = BLOCK_SIZE * 64;
const int COUNT = 1000;

typedef float PixelType;


Cuda::Array2D<float>::Texture tex;



__global__ void
throughput_linear_kernel(Cuda::DeviceMemoryPitched2D<PixelType>::KernelData src_linear,
			 Cuda::DeviceMemoryPitched2D<PixelType>::KernelData dst)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int i = x + y * dst.stride[0];
  dst.data[i] = src_linear.data[i];
}

__global__ void
throughput_array_kernel(Cuda::DeviceMemoryPitched2D<PixelType>::KernelData dst)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int i = x + y * dst.stride[0];
  dst.data[i] = tex2D(tex, x, y);
}

/**
   Compute gigabytes per second.
   @param size edge length of square (i.e., size*size pixels in total)
   @param ms time in milliseconds
*/
float
gbps(float ms)
{
  int bytes = SIZE * SIZE * sizeof(PixelType);
  return bytes * 1000.0 / ms / (1 << 30);
}

void
throughput()
{
  Cuda::Event event0, event1, event2;
  Cuda::DeviceMemoryPitched2D<PixelType> src_linear(SIZE, SIZE);
  Cuda::Array2D<PixelType> src_array(SIZE, SIZE);
  Cuda::DeviceMemoryPitched2D<PixelType> dst(SIZE, SIZE);
  
  Cuda::copy(src_linear, (PixelType)0);
  Cuda::copy(src_array, src_linear);
  src_array.bindTexture(tex);

  dim3 gridDim(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  Cuda::Event t0, t1, t2;
  t0.record();

  for(int i = COUNT; i--;)
    throughput_linear_kernel<<<gridDim, blockDim>>>(src_linear, dst);

  t1.record();

  for(int i = COUNT; i--;)
    throughput_array_kernel<<<gridDim, blockDim>>>(dst);

  t2.record();
  t2.synchronize();
  printf("throughput  linear->linear: %f GB/sec\n", gbps((t1 - t0) / COUNT));
  printf("throughput texture->linear: %f GB/sec\n", gbps((t2 - t1) / COUNT));
}

int
main()
{
  throughput();
  return 0;
}
