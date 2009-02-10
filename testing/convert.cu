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

#include <stdio.h>

#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/devicememorylinear.hpp>

#include <cudatemplates/convert.hpp>


// conversion from float to int:

/*
template <class Type1, class Type2>
__global__ void convert_type_nocheck_int_float_2_kernel(Type1 dst, Type2 src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

template <class Type1, class Type2>
__global__ void convert_type_check_int_float_2_kernel(Type1 dst, Type2 src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}
*/


using namespace std;


int
main()
{
  const size_t SIZE = 5;
  const Cuda::Size<2> size(SIZE, SIZE);

  Cuda::HostMemoryHeap<float, 2> obj1(size);
  Cuda::Size<2> i;

  for(i[1] = SIZE; i[1]--;)
    for(i[0] = SIZE; i[0]--;)
      obj1[i] = random() / 65536.0;

  Cuda::HostMemoryHeap<double, 2> obj2(obj1);

  Cuda::DeviceMemoryLinear<float, 2> obj3(obj1);
  Cuda::DeviceMemoryLinear<int, 2> obj4(size);
  Cuda::HostMemoryHeap<int, 2> obj5(size);
  copy(obj4, obj3);
  copy(obj5, obj4);

  for(i[1] = 0; i[1] < SIZE; ++i[1]) {
    for(i[0] = 0; i[0] < SIZE; ++i[0])
      printf("%9.3f = %5d  ", obj1[i], obj5[i]);

    printf("\n");
  }

  return 0;
}
