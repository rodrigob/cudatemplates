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

#include <assert.h>
#include <stdio.h>

#include <cudatemplates/devicememorymapped.hpp>
#include <cudatemplates/event.hpp>
#include <cudatemplates/hostmemorylocked.hpp>


const int BLOCK_SIZE =  256;
const int NUM_BLOCKS = 1024;
const int COUNT      = 1000;


__global__ void
kernel(int *p1, int *p2)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  p2[i] = p1[i];
}

float
gbps(float ms)
{
  int bytes = BLOCK_SIZE * NUM_BLOCKS * sizeof(int) * 2;
  return bytes * 1000.0 / ms / (1 << 30);
}

int
main()
{
  cudaSetDeviceFlags(cudaDeviceMapHost);
  Cuda::Size<1> size(1 << 20);
  Cuda::HostMemoryLocked<int, 1> h_data1(size, cudaHostAllocMapped);
  Cuda::HostMemoryLocked<int, 1> h_data2(size, cudaHostAllocMapped);
  Cuda::DeviceMemoryMapped<int, 1> d_data1(h_data1);
  Cuda::DeviceMemoryMapped<int, 1> d_data2(h_data2);
  dim3 gridDim(NUM_BLOCKS);
  dim3 blockDim(BLOCK_SIZE);

  for(int i = BLOCK_SIZE * NUM_BLOCKS; i--;)
    h_data1[i] = rand();

  Cuda::Event t0, t1;
  t0.record();

  for(int i = COUNT; i--;)
    kernel<<<gridDim, blockDim>>>(d_data1.getBuffer(), d_data2.getBuffer());

  t1.record();
  t1.synchronize();

  for(int i = BLOCK_SIZE * NUM_BLOCKS; i--;)
    assert(h_data2[i] == h_data1[i]);

  printf("%f GB / sec\n", (t1 - t0) / COUNT);
}
