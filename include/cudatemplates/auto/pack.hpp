/*
  NOTE: THIS FILE HAS BEEN CREATED AUTOMATICALLY,
  ANY CHANGES WILL BE OVERWRITTEN WITHOUT NOTICE!
*/

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

#ifndef CUDA_PACK_AUTO_H
#define CUDA_PACK_AUTO_H


namespace Cuda {


template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_12(VectorType *dst, const ScalarType *src1, const ScalarType *src2, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int dst_ofs = x;
  int src_ofs = x;
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_12(ScalarType *dst1, ScalarType *dst2, const VectorType *src, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int dst_ofs = x;
  int src_ofs = x;
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
}

template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_13(VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int dst_ofs = x;
  int src_ofs = x;
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  vec.z = src3[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_13(ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, const VectorType *src, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int dst_ofs = x;
  int src_ofs = x;
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
  dst3[dst_ofs] = vec.z;
}

template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_14(VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, const ScalarType *src4, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int dst_ofs = x;
  int src_ofs = x;
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  vec.z = src3[src_ofs];
  vec.w = src4[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_14(ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, ScalarType *dst4, const VectorType *src, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int dst_ofs = x;
  int src_ofs = x;
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
  dst3[dst_ofs] = vec.z;
  dst4[dst_ofs] = vec.w;
}

template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_22(VectorType *dst, const ScalarType *src1, const ScalarType *src2, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int dst_ofs = x + y * dst_stride[0];
  int src_ofs = x + y * src_stride[0];
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_22(ScalarType *dst1, ScalarType *dst2, const VectorType *src, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int dst_ofs = x + y * dst_stride[0];
  int src_ofs = x + y * src_stride[0];
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
}

template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_23(VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int dst_ofs = x + y * dst_stride[0];
  int src_ofs = x + y * src_stride[0];
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  vec.z = src3[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_23(ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, const VectorType *src, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int dst_ofs = x + y * dst_stride[0];
  int src_ofs = x + y * src_stride[0];
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
  dst3[dst_ofs] = vec.z;
}

template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_24(VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, const ScalarType *src4, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int dst_ofs = x + y * dst_stride[0];
  int src_ofs = x + y * src_stride[0];
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  vec.z = src3[src_ofs];
  vec.w = src4[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_24(ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, ScalarType *dst4, const VectorType *src, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int dst_ofs = x + y * dst_stride[0];
  int src_ofs = x + y * src_stride[0];
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
  dst3[dst_ofs] = vec.z;
  dst4[dst_ofs] = vec.w;
}

template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_32(VectorType *dst, const ScalarType *src1, const ScalarType *src2, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  int dst_ofs = x + y * dst_stride[0] + z * dst_stride[1];
  int src_ofs = x + y * src_stride[0] + z * src_stride[1];
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_32(ScalarType *dst1, ScalarType *dst2, const VectorType *src, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  int dst_ofs = x + y * dst_stride[0] + z * dst_stride[1];
  int src_ofs = x + y * src_stride[0] + z * src_stride[1];
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
}

template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_33(VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  int dst_ofs = x + y * dst_stride[0] + z * dst_stride[1];
  int src_ofs = x + y * src_stride[0] + z * src_stride[1];
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  vec.z = src3[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_33(ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, const VectorType *src, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  int dst_ofs = x + y * dst_stride[0] + z * dst_stride[1];
  int src_ofs = x + y * src_stride[0] + z * src_stride[1];
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
  dst3[dst_ofs] = vec.z;
}

template <class VectorType, class ScalarType>
__global__ void
pack_nocheck_kernel_34(VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, const ScalarType *src4, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  int dst_ofs = x + y * dst_stride[0] + z * dst_stride[1];
  int src_ofs = x + y * src_stride[0] + z * src_stride[1];
  VectorType vec;
  vec.x = src1[src_ofs];
  vec.y = src2[src_ofs];
  vec.z = src3[src_ofs];
  vec.w = src4[src_ofs];
  dst[dst_ofs] = vec;
}

template <class VectorType, class ScalarType>
__global__ void
unpack_nocheck_kernel_34(ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, ScalarType *dst4, const VectorType *src, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  int dst_ofs = x + y * dst_stride[0] + z * dst_stride[1];
  int src_ofs = x + y * src_stride[0] + z * src_stride[1];
  VectorType vec = src[src_ofs];
  dst1[dst_ofs] = vec.x;
  dst2[dst_ofs] = vec.y;
  dst3[dst_ofs] = vec.z;
  dst4[dst_ofs] = vec.w;
}

template <class VectorType, class ScalarType, unsigned Dim>
struct PackKernel
{
};

template <class VectorType, class ScalarType>
struct PackKernel<VectorType, ScalarType, 1>
{
  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
  {
    pack_nocheck_kernel_12<<<gridDim, blockDim>>>(dst, src1, src2, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, const VectorType *src, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
  {
    unpack_nocheck_kernel_12<<<gridDim, blockDim>>>(dst1, dst2, src, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
  {
    pack_nocheck_kernel_13<<<gridDim, blockDim>>>(dst, src1, src2, src3, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, const VectorType *src, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
  {
    unpack_nocheck_kernel_13<<<gridDim, blockDim>>>(dst1, dst2, dst3, src, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, const ScalarType *src4, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
  {
    pack_nocheck_kernel_14<<<gridDim, blockDim>>>(dst, src1, src2, src3, src4, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, ScalarType *dst4, const VectorType *src, CUDA_KERNEL_SIZE(1) dst_size, CUDA_KERNEL_SIZE(1) dst_stride, CUDA_KERNEL_SIZE(1) src_size, CUDA_KERNEL_SIZE(1) src_stride)
  {
    unpack_nocheck_kernel_14<<<gridDim, blockDim>>>(dst1, dst2, dst3, dst4, src, dst_size, dst_stride, src_size, src_stride);
  }
};

template <class VectorType, class ScalarType>
struct PackKernel<VectorType, ScalarType, 2>
{
  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
  {
    pack_nocheck_kernel_22<<<gridDim, blockDim>>>(dst, src1, src2, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, const VectorType *src, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
  {
    unpack_nocheck_kernel_22<<<gridDim, blockDim>>>(dst1, dst2, src, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
  {
    pack_nocheck_kernel_23<<<gridDim, blockDim>>>(dst, src1, src2, src3, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, const VectorType *src, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
  {
    unpack_nocheck_kernel_23<<<gridDim, blockDim>>>(dst1, dst2, dst3, src, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, const ScalarType *src4, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
  {
    pack_nocheck_kernel_24<<<gridDim, blockDim>>>(dst, src1, src2, src3, src4, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, ScalarType *dst4, const VectorType *src, CUDA_KERNEL_SIZE(2) dst_size, CUDA_KERNEL_SIZE(2) dst_stride, CUDA_KERNEL_SIZE(2) src_size, CUDA_KERNEL_SIZE(2) src_stride)
  {
    unpack_nocheck_kernel_24<<<gridDim, blockDim>>>(dst1, dst2, dst3, dst4, src, dst_size, dst_stride, src_size, src_stride);
  }
};

template <class VectorType, class ScalarType>
struct PackKernel<VectorType, ScalarType, 3>
{
  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
  {
    pack_nocheck_kernel_32<<<gridDim, blockDim>>>(dst, src1, src2, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, const VectorType *src, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
  {
    unpack_nocheck_kernel_32<<<gridDim, blockDim>>>(dst1, dst2, src, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
  {
    pack_nocheck_kernel_33<<<gridDim, blockDim>>>(dst, src1, src2, src3, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, const VectorType *src, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
  {
    unpack_nocheck_kernel_33<<<gridDim, blockDim>>>(dst1, dst2, dst3, src, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void pack_nocheck(dim3 gridDim, dim3 blockDim, VectorType *dst, const ScalarType *src1, const ScalarType *src2, const ScalarType *src3, const ScalarType *src4, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
  {
    pack_nocheck_kernel_34<<<gridDim, blockDim>>>(dst, src1, src2, src3, src4, dst_size, dst_stride, src_size, src_stride);
  }

  static inline void unpack_nocheck(dim3 gridDim, dim3 blockDim, ScalarType *dst1, ScalarType *dst2, ScalarType *dst3, ScalarType *dst4, const VectorType *src, CUDA_KERNEL_SIZE(3) dst_size, CUDA_KERNEL_SIZE(3) dst_stride, CUDA_KERNEL_SIZE(3) src_size, CUDA_KERNEL_SIZE(3) src_stride)
  {
    unpack_nocheck_kernel_34<<<gridDim, blockDim>>>(dst1, dst2, dst3, dst4, src, dst_size, dst_stride, src_size, src_stride);
  }
};

template<class VectorType, class ScalarType, unsigned Dim>
void
pack(DeviceMemory<VectorType, Dim> &dst,
     const DeviceMemory<ScalarType, Dim> &src1,
     const DeviceMemory<ScalarType, Dim> &src2)
{
  // TODO: size check
  Size<Dim> dst_ofs, size(dst.size);
  // dst.checkBounds(dst_ofs, size);
  dim3 gridDim, blockDim;
  bool aligned;
  size_t dofs;
  Size<Dim> rmin, rmax;
  dst.getExecutionConfiguration(gridDim, blockDim, aligned, dofs, rmin, rmax, dst_ofs, size);
  // typename DeviceMemory<Type, Dim>::KernelData kdst(dst);
  // kdst.data += dofs;

  if(aligned)
    PackKernel<VectorType, ScalarType, Dim>::pack_nocheck(gridDim, blockDim, dst.getBuffer(), src1.getBuffer(), src2.getBuffer(), dst.size, dst.stride, src1.size, src1.stride);
  else
    abort();  // pack_check_kernel<<<gridDim, blockDim>>>(kdst, val, rmin, rmax);

  CUDA_CHECK_LAST;
}

template<class VectorType, class ScalarType, unsigned Dim>
void
unpack(DeviceMemory<ScalarType, Dim> &dst1,
       DeviceMemory<ScalarType, Dim> &dst2,
       const DeviceMemory<VectorType, Dim> &src)
{
  // TODO: size check
  Size<Dim> dst_ofs, size(src.size);
  // dst.checkBounds(dst_ofs, size);
  dim3 gridDim, blockDim;
  bool aligned;
  size_t dofs;
  Size<Dim> rmin, rmax;
  src.getExecutionConfiguration(gridDim, blockDim, aligned, dofs, rmin, rmax, dst_ofs, size);
  // typename DeviceMemory<Type, Dim>::KernelData kdst(dst);
  // kdst.data += dofs;

  if(aligned)
    PackKernel<VectorType, ScalarType, Dim>::unpack_nocheck(gridDim, blockDim, dst1.getBuffer(), dst2.getBuffer(), src.getBuffer(), dst1.size, dst1.stride, src.size, src.stride);
  else
    abort();  // unpack_check_kernel<<<gridDim, blockDim>>>(kdst, val, rmin, rmax);

  CUDA_CHECK_LAST;
}

template<class VectorType, class ScalarType, unsigned Dim>
void
pack(DeviceMemory<VectorType, Dim> &dst,
     const DeviceMemory<ScalarType, Dim> &src1,
     const DeviceMemory<ScalarType, Dim> &src2,
     const DeviceMemory<ScalarType, Dim> &src3)
{
  // TODO: size check
  Size<Dim> dst_ofs, size(dst.size);
  // dst.checkBounds(dst_ofs, size);
  dim3 gridDim, blockDim;
  bool aligned;
  size_t dofs;
  Size<Dim> rmin, rmax;
  dst.getExecutionConfiguration(gridDim, blockDim, aligned, dofs, rmin, rmax, dst_ofs, size);
  // typename DeviceMemory<Type, Dim>::KernelData kdst(dst);
  // kdst.data += dofs;

  if(aligned)
    PackKernel<VectorType, ScalarType, Dim>::pack_nocheck(gridDim, blockDim, dst.getBuffer(), src1.getBuffer(), src2.getBuffer(), src3.getBuffer(), dst.size, dst.stride, src1.size, src1.stride);
  else
    abort();  // pack_check_kernel<<<gridDim, blockDim>>>(kdst, val, rmin, rmax);

  CUDA_CHECK_LAST;
}

template<class VectorType, class ScalarType, unsigned Dim>
void
unpack(DeviceMemory<ScalarType, Dim> &dst1,
       DeviceMemory<ScalarType, Dim> &dst2,
       DeviceMemory<ScalarType, Dim> &dst3,
       const DeviceMemory<VectorType, Dim> &src)
{
  // TODO: size check
  Size<Dim> dst_ofs, size(src.size);
  // dst.checkBounds(dst_ofs, size);
  dim3 gridDim, blockDim;
  bool aligned;
  size_t dofs;
  Size<Dim> rmin, rmax;
  src.getExecutionConfiguration(gridDim, blockDim, aligned, dofs, rmin, rmax, dst_ofs, size);
  // typename DeviceMemory<Type, Dim>::KernelData kdst(dst);
  // kdst.data += dofs;

  if(aligned)
    PackKernel<VectorType, ScalarType, Dim>::unpack_nocheck(gridDim, blockDim, dst1.getBuffer(), dst2.getBuffer(), dst3.getBuffer(), src.getBuffer(), dst1.size, dst1.stride, src.size, src.stride);
  else
    abort();  // unpack_check_kernel<<<gridDim, blockDim>>>(kdst, val, rmin, rmax);

  CUDA_CHECK_LAST;
}

template<class VectorType, class ScalarType, unsigned Dim>
void
pack(DeviceMemory<VectorType, Dim> &dst,
     const DeviceMemory<ScalarType, Dim> &src1,
     const DeviceMemory<ScalarType, Dim> &src2,
     const DeviceMemory<ScalarType, Dim> &src3,
     const DeviceMemory<ScalarType, Dim> &src4)
{
  // TODO: size check
  Size<Dim> dst_ofs, size(dst.size);
  // dst.checkBounds(dst_ofs, size);
  dim3 gridDim, blockDim;
  bool aligned;
  size_t dofs;
  Size<Dim> rmin, rmax;
  dst.getExecutionConfiguration(gridDim, blockDim, aligned, dofs, rmin, rmax, dst_ofs, size);
  // typename DeviceMemory<Type, Dim>::KernelData kdst(dst);
  // kdst.data += dofs;

  if(aligned)
    PackKernel<VectorType, ScalarType, Dim>::pack_nocheck(gridDim, blockDim, dst.getBuffer(), src1.getBuffer(), src2.getBuffer(), src3.getBuffer(), src4.getBuffer(), dst.size, dst.stride, src1.size, src1.stride);
  else
    abort();  // pack_check_kernel<<<gridDim, blockDim>>>(kdst, val, rmin, rmax);

  CUDA_CHECK_LAST;
}

template<class VectorType, class ScalarType, unsigned Dim>
void
unpack(DeviceMemory<ScalarType, Dim> &dst1,
       DeviceMemory<ScalarType, Dim> &dst2,
       DeviceMemory<ScalarType, Dim> &dst3,
       DeviceMemory<ScalarType, Dim> &dst4,
       const DeviceMemory<VectorType, Dim> &src)
{
  // TODO: size check
  Size<Dim> dst_ofs, size(src.size);
  // dst.checkBounds(dst_ofs, size);
  dim3 gridDim, blockDim;
  bool aligned;
  size_t dofs;
  Size<Dim> rmin, rmax;
  src.getExecutionConfiguration(gridDim, blockDim, aligned, dofs, rmin, rmax, dst_ofs, size);
  // typename DeviceMemory<Type, Dim>::KernelData kdst(dst);
  // kdst.data += dofs;

  if(aligned)
    PackKernel<VectorType, ScalarType, Dim>::unpack_nocheck(gridDim, blockDim, dst1.getBuffer(), dst2.getBuffer(), dst3.getBuffer(), dst4.getBuffer(), src.getBuffer(), dst1.size, dst1.stride, src.size, src.stride);
  else
    abort();  // unpack_check_kernel<<<gridDim, blockDim>>>(kdst, val, rmin, rmax);

  CUDA_CHECK_LAST;
}

}  // namespace Cuda


#endif
