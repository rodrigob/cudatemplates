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

#ifndef CUDA_CONVERT_H
#define CUDA_CONVERT_H


#include <cudatemplates/copy.hpp>
#include <cudatemplates/hostmemory.hpp>


#ifdef __CUDACC__

template <class Type1, class Type2>
__global__ void convert_type_1d_nocheck_kernel(Type1 dst, Type2 src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

template <class Type1, class Type2>
__global__ void convert_type_1d_check_kernel(Type1 dst, Type2 src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

template <class Type1, class Type2>
__global__ void convert_type_2d_nocheck_kernel(Type1 dst, Type2 src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

template <class Type1, class Type2>
__global__ void convert_type_2d_check_kernel(Type1 dst, Type2 src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if((x < dst.size[0]) && (y < dst.size[1]))
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

template <class Type1, class Type2>
__global__ void convert_type_3d_nocheck_kernel(Type1 dst, Type2 src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

template <class Type1, class Type2>
__global__ void convert_type_3d_check_kernel(Type1 dst, Type2 src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if((x < dst.size[0]) && (y < dst.size[1]) && (z < dst.size[2]))
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

#endif

namespace Cuda {

/**
   Convert data in host memory.
   @param dst destination pointer
   @param src source pointer
*/
template<class Type1, class Type2, unsigned Dim>
void
copy(HostMemory<Type1, Dim> &dst, const HostMemory<Type2, Dim> &src)
{
  CUDA_CHECK_SIZE;
  Cuda::Iterator<Dim> src_end = src.end();

  for(Cuda::Iterator<Dim> i = src.begin(); i != src_end; ++i)
    dst[i] = src[i];
}

#ifdef __CUDACC__

static inline int divup(int a, int b)
{
  return (a + b - 1) / b;
}

/**
   Convert data in device memory.
   The generic implementation is undefined, all behaviour is implemented in
   partial specializations for each dimension.
   @param dst destination pointer
   @param src source pointer
*/
template<class Type1, class Type2, unsigned Dim>
void
copy(DeviceMemory<Type1, Dim> &dst, const DeviceMemory<Type2, Dim> &src);

/**
   Convert one-dimensional data in device memory.
   Since this function calls a CUDA kernel, it is only available if the file
   from which this function is called is compiled by nvcc.
   @param dst destination
   @param src source
*/
template<class Type1, class Type2>
void
copy(DeviceMemory<Type1, 1> &dst, const DeviceMemory<Type2, 1> &src)
{
  CUDA_CHECK_SIZE;
  dim3 blockDim = dim3(64, 1, 1);
  dim3 gridDim = dim3(divup(dst.size[0], blockDim.x), 1, 1);
    
  typename DeviceMemory<Type1, 1>::KernelData kdst(dst);
  typename DeviceMemory<Type2, 1>::KernelData ksrc(src);

  if((dst.size[0] % blockDim.x) == 0)
    convert_type_1d_nocheck_kernel<<<gridDim, blockDim>>>(kdst, ksrc);
  else
    convert_type_1d_check_kernel<<<gridDim, blockDim>>>(kdst, ksrc);
}

/**
   Convert two-dimensional data in device memory.
   Since this function calls a CUDA kernel, it is only available if the file
   from which this function is called is compiled by nvcc.
   @param dst destination
   @param src source
*/
template<class Type1, class Type2>
void
copy(DeviceMemory<Type1, 2> &dst, const DeviceMemory<Type2, 2> &src)
{
  CUDA_CHECK_SIZE;
  dim3 blockDim = dim3(32, 4, 1);
  dim3 gridDim = dim3(divup(dst.size[0], blockDim.x),
		      divup(dst.size[1], blockDim.y),
		      1);
    
  typename DeviceMemory<Type1, 2>::KernelData kdst(dst);
  typename DeviceMemory<Type2, 2>::KernelData ksrc(src);

  if(((dst.size[0] % blockDim.x) == 0) &&
     ((dst.size[1] % blockDim.y) == 0))
    convert_type_2d_nocheck_kernel<<<gridDim, blockDim>>>(kdst, ksrc);
  else
    convert_type_2d_check_kernel<<<gridDim, blockDim>>>(kdst, ksrc);
}

/**
   Convert three-dimensional data in device memory.
   Since this function calls a CUDA kernel, it is only available if the file
   from which this function is called is compiled by nvcc.
   @param dst destination
   @param src source
*/
template<class Type1, class Type2>
void
copy(DeviceMemory<Type1, 3> &dst, const DeviceMemory<Type2, 3> &src)
{
  CUDA_CHECK_SIZE;
  dim3 blockDim = dim3(8, 8, 8);
  dim3 gridDim = dim3(divup(dst.size[0], blockDim.x),
		      divup(dst.size[1], blockDim.y),
		      divup(dst.size[2], blockDim.z));

  typename DeviceMemory<Type1, 3>::KernelData kdst(dst);
  typename DeviceMemory<Type2, 3>::KernelData ksrc(src);

  if(((dst.size[0] % blockDim.x) == 0) &&
     ((dst.size[1] % blockDim.y) == 0) &&
     ((dst.size[2] % blockDim.z) == 0))
    convert_type_3d_nocheck_kernel<<<gridDim, blockDim>>>(kdst, ksrc);
  else
    convert_type_3d_check_kernel<<<gridDim, blockDim>>>(kdst, ksrc);
}

/**
   Force instantiation of template kernels.
   nvcc gets confused if it doesn't see at least one instantiation of template
   kernels. This dummy class creates one.
*/
struct DummyInstantiateTemplateKernels
{
  DummyInstantiateTemplateKernels()
  {
    dim3 gridDim(1, 1, 1), blockDim(1, 1, 1);

    typename DeviceMemory<int, 1>::KernelData data1;
    convert_type_1d_nocheck_kernel<<<gridDim, blockDim>>>(data1, data1);
    convert_type_1d_check_kernel<<<gridDim, blockDim>>>(data1, data1);

    typename DeviceMemory<int, 2>::KernelData data2;
    convert_type_2d_nocheck_kernel<<<gridDim, blockDim>>>(data2, data2);
    convert_type_2d_check_kernel<<<gridDim, blockDim>>>(data2, data2);

    typename DeviceMemory<int, 3>::KernelData data3;
    convert_type_3d_nocheck_kernel<<<gridDim, blockDim>>>(data3, data3);
    convert_type_3d_check_kernel<<<gridDim, blockDim>>>(data3, data3);
  }
};

#endif

/**
   Convert data in host memory.
   @param dst generic destination pointer
   @param src generic source pointer
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be converted
*/
template<class Type1, class Type2, unsigned Dim>
void
copy(HostMemory<Type1, Dim> &dst, const HostMemory<Type2, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  check_bounds(dst, src, dst_ofs, src_ofs, size);
  Cuda::Iterator<Dim> src_begin(src_ofs, Cuda::Size<Dim>(src_ofs + size));
  Cuda::Iterator<Dim> src_end = src_begin;
  src_end.setEnd();
  Cuda::Iterator<Dim> dst_begin(dst_ofs, Cuda::Size<Dim>(dst_ofs + size));

  for(Cuda::Iterator<Dim> i = src_begin, j = dst_begin; i != src_end; ++i, ++j)
    dst[j] = src[i];
}

}  // namespace Cuda


#endif
