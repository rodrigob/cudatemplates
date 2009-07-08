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
writeImageMem(unsigned long long address, int xsize, int ysize)
{
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  PixelType* xaddress = (PixelType*)address;

  if (i<xsize && j< ysize)
  {

    *(xaddress+i+j*xsize) = 0.5;
    
  }
}

__global__ void 
findPatternStartAdress(Cuda::DeviceMemory<PixelType, 2>::KernelData pattern, 
                       int searchSpaceBytes, int iteration,
                       int xsize, int ysize,
                       Cuda::DeviceMemoryLinear1D<unsigned long long>::KernelData mct
                       )
{
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  PixelType* address = pattern.data-500000;

  if (i==0 && j==0) mct.data[1] = (unsigned long long)(address+iteration);

  bool matchp = true;


  bool mismatch = false;
  if (i<xsize && j<ysize) 
  {

    address += iteration+(i+j*xsize);
    PixelType value = *address;
    PixelType pval = *(pattern.data+i+j*xsize);
    if (!isnan(value) && !isinf(value))
    {
      if (pval != value) 
      {
        mismatch = true;
      }
    }
  }

  __syncthreads();
  if (__any(mismatch))
  {
    matchp = false;
  }
  __syncthreads();
  if ( threadIdx.x == 0 && threadIdx.y == 0)
  {
    unsigned long long ov;

    if (!matchp) 
      ov = atomicAdd(&mct.data[0], 
                     1ull);
      
  }
  //write test
  //*adress = 0;

  
}

void find_crazyAddress(Cuda::HostMemoryHeap2D<PixelType> pattern, 
                      Cuda::Size<2> size, 
                      Cuda::HostMemoryHeap1D<unsigned long long> mcth, 
                      Cuda::DeviceMemoryLinear1D<unsigned long long> mct)
{

  dim3 gridDim(div_up(size[0],BLOCK_SIZE), div_up(size[1], BLOCK_SIZE));
  dim3 gridDimRed(1,1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  printf("gridDim %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
  printf("size1 %lu x %lu\n", size[0], size[1]);    
  Cuda::DeviceMemoryLinear2D<PixelType> d_pattern(size);
  Cuda::copy(d_pattern, pattern);
  printf("size2 %lu x %lu\n", size[0], size[1]);    

 cudaDeviceProp deviceProp;
 cudaGetDeviceProperties(&deviceProp, 0);
 printf("\nDevice %d: \"%s\"\n", 0, deviceProp.name);
 printf("  Total amount of global memory:                 %lu bytes\n", deviceProp.totalGlobalMem);
 
 // TODO : find a good search strategy
 unsigned long iterations = deviceProp.totalGlobalMem - (size[0]*size[1]);
 printf("necessary iterations: %lu \n", iterations);
 
 // TODO : do as 

 iterations = 500001;
 for(unsigned long i = 0; i < iterations; i++)
 {
   if (i%10000 ==0 )
   {
     printf("iteration %lu \n",i);
   }

   mcth[0] = 0;
   copy(mct, mcth);

   findPatternStartAdress<<<gridDimRed,blockDim>>>(d_pattern, deviceProp.totalGlobalMem, i, size[0], size[1], mct);

   copy(mcth, mct);
   //printf("result: %lu\n", mcth[0]);
   
   if (mcth[0] == 0)   
   {
     printf("Likely found pattern at position %lu \n", i);
     copy(mct, mcth);
     
     findPatternStartAdress<<<gridDim,blockDim>>>(d_pattern, deviceProp.totalGlobalMem, i, size[0], size[1], mct);
     copy(mcth, mct);

     if (mcth[0] == 0)
     {
       printf("Really found pattern at position %lu, address  %llx\n", i, mcth[1]);
       writeImageMem<<<gridDim,blockDim>>>(mcth[1], size[0], size[1]);
       //break;
     }
   }
 }

}

//========================================================================
// End of file
//========================================================================
// Local Variables:
// mode: c++
// c-basic-offset: 2
// eval: (c-set-offset 'substatement-open 0)
// eval: (c-set-offset 'case-label '+)
// eval: (c-set-offset 'statement 'c-lineup-runin-statements)
// eval: (setq indent-tabs-mode nil)
// End:
//========================================================================
