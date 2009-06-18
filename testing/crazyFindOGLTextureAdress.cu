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

#define GL_GLEXT_PROTOTYPES

#include <assert.h>
#include <GL/glew.h>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/opengl/bufferobject.hpp>
#include <cudatemplates/opengl/copy.hpp>
#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/devicememorylinear.hpp>

const int BLOCK_SIZE = 8;
typedef float PixelType;

/**
   Integer division (rounding up the result).
*/
static inline int
div_up(int x, int y)
{
  return (x + y - 1) / y;
}

__global__ void 
findPatternStartAdress(Cuda::DeviceMemory<PixelType, 2>::KernelData pattern, int searchSpaceBytes, int iteration)
{
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  PixelType* adress = 0;

  adress += iteration+(i+i*j);

  PixelType value = *adress;

  //write test
  *adress = 0;

  
}

void find_crazyAdress(Cuda::HostMemoryHeap2D<PixelType> pattern, Cuda::Size<2> size)
{

  dim3 gridDim(size[0] / BLOCK_SIZE, size[1] / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

  Cuda::DeviceMemoryLinear2D<PixelType> d_pattern(size);
  Cuda::copy(d_pattern, pattern);

 cudaDeviceProp deviceProp;
 cudaGetDeviceProperties(&deviceProp, 0);
 printf("\nDevice %d: \"%s\"\n", 0, deviceProp.name);
 printf("  Total amount of global memory:                 %d bytes\n", deviceProp.totalGlobalMem);
 
 // TODO : find a good search strategy
 int iterations = deviceProp.totalGlobalMem - (size[0]*size[1]);
 printf("necessary iterations: %d \n", iterations);
 
 // TODO : do as 
 for(int i = 0; i < iterations; i++)
 {
 findPatternStartAdress<<<gridDim,blockDim>>>(d_pattern, deviceProp.totalGlobalMem, i);
 }

}