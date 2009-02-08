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


namespace Cuda {


template <class Type1, class Type2>
void convert_type_nocheck(Type1 &dst, Type2 &src, dim3 gridDim, dim3 blockDim);

template <class Type1, class Type2>
void convert_type_check(Type1 &dst, Type2 &src, dim3 gridDim, dim3 blockDim);

static inline int divup(int a, int b)
{
  return (a + b - 1) / b;
}


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

/**
   Convert data in device memory.
   @param dst destination pointer
   @param src source pointer
*/
template<class Type1, class Type2, unsigned Dim>
void
copy(DeviceMemory<Type1, Dim> &dst, const DeviceMemory<Type2, Dim> &src)
{
  CUDA_CHECK_SIZE;
  CUDA_STATIC_ASSERT(Dim >= 1);
  CUDA_STATIC_ASSERT(Dim <= 3);

  dim3 blockDim, gridDim;
  typename DeviceMemory<Type1, Dim>::KernelData kdst(dst);
  typename DeviceMemory<Type2, Dim>::KernelData ksrc(src);

  switch(Dim) {
  case 1:
    blockDim = dim3(64, 1, 1);
    gridDim = dim3(divup(dst.size[0], blockDim.x), 1, 1);
    
    if((dst.size[0] % blockDim.x) == 0)
      convert_type_nocheck(kdst, ksrc, gridDim, blockDim);
    else
      convert_type_check(kdst, ksrc, gridDim, blockDim);

    break;

  case 2:
    blockDim = dim3(32, 4, 1);
    gridDim = dim3(divup(dst.size[0], blockDim.x),
		   divup(dst.size[1], blockDim.y),
		   1);

    if(((dst.size[0] % blockDim.x) == 0) &&
       ((dst.size[1] % blockDim.y) == 0))
      convert_type_nocheck(kdst, ksrc, gridDim, blockDim);
    else
      convert_type_check(kdst, ksrc, gridDim, blockDim);

    break;

  case 3:
    blockDim = dim3(8, 8, 8);
    gridDim = dim3(divup(dst.size[0], blockDim.x),
		   divup(dst.size[1], blockDim.y),
		   divup(dst.size[2], blockDim.z));

    if(((dst.size[0] % blockDim.x) == 0) &&
       ((dst.size[1] % blockDim.y) == 0) &&
       ((dst.size[2] % blockDim.z) == 0))
      convert_type_nocheck(kdst, ksrc, gridDim, blockDim);
    else
      convert_type_check(kdst, ksrc, gridDim, blockDim);
  }
}

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
