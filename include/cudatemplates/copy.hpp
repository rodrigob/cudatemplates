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

#ifndef CUDA_COPY_H
#define CUDA_COPY_H


#include <cudatemplates/array.hpp>
#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/dimension.hpp>
#include <cudatemplates/hostmemory.hpp>
#include <cudatemplates/hostmemoryreference.hpp>
#include <cudatemplates/staticassert.hpp>
#include <cudatemplates/symbol.hpp>


/**
   Using offsets in cudaMemcpy3D only works for arrays,
   although the documentation doesn't say so.
   By default, a workaround is enabled, but we leave this in the code
   to check if this is fixed in a later version.
*/
#define CUDA_USE_OFFSET 0

/*
  There is a possible range checking bug in cudaMemcpy3D,
  see http://forums.nvidia.com/index.php?showtopic=73497.
*/


#define CUDA_CHECK_SIZE if(dst.size != src.size) CUDA_ERROR("size mismatch")


#ifdef __CUDACC__

template <class Type1, class Type2>
__global__ void copy_constant_nocheck_kernel(Type1 dst, Type2 val, Cuda::Dimension<1> dummy)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = val;
}

template <class Type1, class Type2>
__global__ void copy_constant_check_kernel(Type1 dst, Type2 val, Cuda::Size<1> rmin, Cuda::Size<1> rmax)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if((x >= rmin[0]) && (x < rmax[0]))
    dst.data[x] = val;
}

template <class Type1, class Type2>
__global__ void copy_constant_nocheck_kernel(Type1 dst, Type2 val, Cuda::Dimension<2> dummy)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = val;
}

template <class Type1, class Type2>
__global__ void copy_constant_check_kernel(Type1 dst, Type2 val, Cuda::Size<2> rmin, Cuda::Size<2> rmax)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if((x >= rmin[0]) && (x < rmax[0]) && (y >= rmin[1]) && (y < rmax[1]))
    dst.data[x + y * dst.stride[0]] = val;
}

template <class Type1, class Type2>
__global__ void copy_constant_nocheck_kernel(Type1 dst, Type2 val, Cuda::Dimension<3> dummy)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = val;
}

template <class Type1, class Type2>
__global__ void copy_constant_check_kernel(Type1 dst, Type2 val, Cuda::Size<3> rmin, Cuda::Size<3> rmax)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if((x >= rmin[0]) && (x < rmax[0]) && (y >= rmin[1]) && (y < rmax[1]) && (z >= rmin[2]) && (z < rmax[2]))
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = val;

  dst.data[0] = 12345678;
}

#endif  // __CUDACC__


namespace Cuda {

typedef enum {
  BORDER_CLAMP/*,
  BORDER_MIRROR,
  BORDER_REPEAT*/
} border_t;

template<class Type1, class Type2, unsigned Dim>
static void
check_bounds(const Layout<Type1, Dim> &dst, const Layout<Type2, Dim> &src,
	     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  for(size_t i = Dim; i--;) {
    if((dst_ofs[i] >= dst.size[i]) ||
       (dst_ofs[i] + size[i] > dst.size[i]) ||
       (src_ofs[i] >= src.size[i]) ||
       (src_ofs[i] + size[i] > src.size[i]))
      CUDA_ERROR("out of bounds");
  }
}

/**
   Generic copy method for host and/or device memory.
   It is not recommended to call this function directly since its correct
   behaviour depends on the kind parameter (just as the underlying CUDA
   functions).
   @param dst generic destination pointer
   @param src generic source pointer
   @param kind direction of copy
*/
template<class Type, unsigned Dim>
void
copy(Pointer<Type, Dim> &dst, const Pointer<Type, Dim> &src, cudaMemcpyKind kind)
{
  CUDA_STATIC_ASSERT(Dim >= 1);
  CUDA_CHECK_SIZE;

  if(Dim == 1) {
    CUDA_CHECK(cudaMemcpy(dst.getBuffer(), src.getBuffer(), src.getSize() * sizeof(Type), kind));
  }
  else if(Dim == 2) {
    CUDA_CHECK(cudaMemcpy2D(dst.getBuffer(), dst.getPitch(), src.getBuffer(), src.getPitch(),
			    src.size[0] * sizeof(Type), src.size[1], kind));
  }
  else if(Dim >= 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcPtr.ptr = (void *)src.getBuffer();
    p.srcPtr.pitch = src.getPitch();
    p.srcPtr.xsize = src.size[0];
    p.srcPtr.ysize = src.size[1];
    p.dstPtr.ptr = (void *)dst.getBuffer();
    p.dstPtr.pitch = dst.getPitch();
    p.dstPtr.xsize = dst.size[0];
    p.dstPtr.ysize = dst.size[1];
    p.extent.width = src.size[0] * sizeof(Type);  // no CUDA array involved -> width is given in bytes
    p.extent.height = src.size[1];
    p.extent.depth = src.size[2];

    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= src.size[i];

    p.kind = kind;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

/**
   Generic region copy method for host and/or device memory.
   It is not recommended to call this function directly since its correct
   behaviour depends on the kind parameter (just as the underlying CUDA
   functions).
   @param dst generic destination pointer
   @param src generic source pointer
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
   @param kind direction of copy
*/
template<class Type, unsigned Dim>
void
copy(Pointer<Type, Dim> &dst, const Pointer<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size,
     cudaMemcpyKind kind)
{
  CUDA_STATIC_ASSERT(Dim >= 1);
  CUDA_STATIC_ASSERT(Dim <= 3);

  check_bounds(dst, src, dst_ofs, src_ofs, size);

  if(Dim == 1) {
    CUDA_CHECK(cudaMemcpy(dst.getBuffer() + dst_ofs[0], src.getBuffer() + src_ofs[0], size[0] * sizeof(Type), kind));
  }
  else if(Dim == 2) {
    CUDA_CHECK(cudaMemcpy2D(dst.getBuffer() + dst_ofs[0] + dst_ofs[1] * dst.stride[0], dst.getPitch(),
			    src.getBuffer() + src_ofs[0] + src_ofs[1] * src.stride[0], src.getPitch(),
			    size[0] * sizeof(Type), size[1], kind));
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
#if CUDA_USE_OFFSET
    p.srcPos.x = src_ofs[0] * sizeof(Type);
    p.srcPos.y = src_ofs[1];
    p.srcPos.z = src_ofs[2];
    p.srcPtr.ptr = (void *)src.getBuffer();
#else
    p.srcPtr.ptr = (void *)(src.getBuffer() + src_ofs[0] + src_ofs[1] * src.stride[0] + src_ofs[2] * src.stride[1]);
#endif
    p.srcPtr.pitch = src.getPitch();
    p.srcPtr.xsize = src.size[0];
    p.srcPtr.ysize = src.size[1];
#if CUDA_USE_OFFSET
    p.dstPos.x = dst_ofs[0] * sizeof(Type);
    p.dstPos.y = dst_ofs[1];
    p.dstPos.z = dst_ofs[2];
    p.dstPtr.ptr = (void *)dst.getBuffer();
#else
    p.dstPtr.ptr = (void *)(dst.getBuffer() + dst_ofs[0] + dst_ofs[1] * dst.stride[0] + dst_ofs[2] * dst.stride[1]);
#endif
    p.dstPtr.pitch = dst.getPitch();
    p.dstPtr.xsize = dst.size[0];
    p.dstPtr.ysize = dst.size[1];
    p.extent.width = size[0] * sizeof(Type);  // no CUDA array involved -> width is given in bytes
    p.extent.height = size[1];
    p.extent.depth = size[2];
    p.kind = kind;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

/**
   Copy host memory to device memory.
   @param dst destination pointer (device memory)
   @param src source pointer (host memory)
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, const HostMemory<Type, Dim> &src)
{
  copy(dst, src, cudaMemcpyHostToDevice);
}

/**
   Copy region from host memory to device memory.
   @param dst destination pointer (device memory)
   @param src source pointer (host memory)
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, const HostMemory<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  copy(dst, src, dst_ofs, src_ofs, size, cudaMemcpyHostToDevice);
}

/**
   Copy device memory to host memory.
   @param dst destination pointer (host memory)
   @param src source pointer (device memory)
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const DeviceMemory<Type, Dim> &src)
{
  copy(dst, src, cudaMemcpyDeviceToHost);
}

/**
   Copy region from device memory to host memory.
   @param dst destination pointer (host memory)
   @param src source pointer (device memory)
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const DeviceMemory<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  copy(dst, src, dst_ofs, src_ofs, size, cudaMemcpyDeviceToHost);
}

/**
   Copy device memory to device memory.
   @param dst destination pointer (device memory)
   @param src source pointer (device memory)
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, const DeviceMemory<Type, Dim> &src)
{
  copy(dst, src, cudaMemcpyDeviceToDevice);
}

/**
   Copy region from device memory to device memory.
   @param dst destination pointer (device memory)
   @param src source pointer (device memory)
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, const DeviceMemory<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  copy(dst, src, dst_ofs, src_ofs, size, cudaMemcpyDeviceToDevice);
}

/**
   Copy host memory to host memory.
   @param dst destination pointer (host memory)
   @param src source pointer (host memory)
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const HostMemory<Type, Dim> &src)
{
  copy(dst, src, cudaMemcpyHostToHost);
}

/**
   Copy region from host memory to host memory.
   @param dst destination pointer (host memory)
   @param src source pointer (host memory)
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const HostMemory<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  copy(dst, src, dst_ofs, src_ofs, size, cudaMemcpyHostToHost);
}

//------------------------------------------------------------------------------
/**
   Generic copy method from CUDA array to host/device memory.
   It is not recommended to call this function directly since its correct
   behaviour depends on the kind parameter (just as the underlying CUDA
   functions).
   @param dst generic destination pointer
   @param src source CUDA array
   @param kind direction of copy
*/
template<class Type, unsigned Dim>
void
copyFromArray(Pointer<Type, Dim> &dst, const Array<Type, Dim> &src, cudaMemcpyKind kind)
{
  CUDA_STATIC_ASSERT(Dim >= 2);
  CUDA_STATIC_ASSERT(Dim <= 3);
  CUDA_CHECK_SIZE;

  if(Dim == 2) {
    CUDA_CHECK(cudaMemcpy2DFromArray(dst.getBuffer(), dst.getPitch(),
				     src.getArray(), 0, 0,
				     src.size[0] * sizeof(Type), src.size[1], kind));
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcArray = const_cast<cudaArray *>(src.getArray());
    p.srcPtr.pitch = src.getPitch();
    p.srcPtr.xsize = src.size[0];
    p.srcPtr.ysize = src.size[1];
    p.dstPtr.ptr = (void *)dst.getBuffer();
    p.dstPtr.pitch = dst.getPitch();
    p.dstPtr.xsize = dst.size[0];
    p.dstPtr.ysize = dst.size[1];
    p.extent.width = src.size[0];
    p.extent.height = src.size[1];
    p.extent.depth = src.size[2];

    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= src.size[i];

    p.kind = kind;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

/**
   Generic region copy method from CUDA array to host/device memory.
   It is not recommended to call this function directly since its correct
   behaviour depends on the kind parameter (just as the underlying CUDA
   functions).
   @param dst generic destination pointer
   @param src source CUDA array
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
   @param kind direction of copy
*/
template<class Type, unsigned Dim>
void
copyFromArray(Pointer<Type, Dim> &dst, const Array<Type, Dim> &src,
	      const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size,
	      cudaMemcpyKind kind)
{
  CUDA_STATIC_ASSERT(Dim >= 2);
  CUDA_STATIC_ASSERT(Dim <= 3);

  check_bounds(dst, src, dst_ofs, src_ofs, size);

  if(Dim == 2) {
    CUDA_CHECK(cudaMemcpy2DFromArray(dst.getBuffer() + dst_ofs[0] + dst_ofs[1] * dst.stride[0], dst.getPitch(),
				     src.getArray(), src_ofs[0] * sizeof(Type), src_ofs[1],
				     size[0] * sizeof(Type), size[1], kind));
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcArray = const_cast<cudaArray *>(src.getArray());
    p.srcPos.x = src_ofs[0];
    p.srcPos.y = src_ofs[1];
    p.srcPos.z = src_ofs[2];
    p.srcPtr.pitch = src.getPitch();
    p.srcPtr.xsize = src.size[0];
    p.srcPtr.ysize = src.size[1];
#if CUDA_USE_OFFSET
    p.dstPos.x = dst_ofs[0] * sizeof(Type);
    p.dstPos.y = dst_ofs[1];
    p.dstPos.z = dst_ofs[2];
    p.dstPtr.ptr = (void *)dst.getBuffer();
#else
    p.dstPtr.ptr = (void *)(dst.getBuffer() + dst_ofs[0] + dst_ofs[1] * dst.stride[0] + dst_ofs[2] * dst.stride[1]);
#endif
    p.dstPtr.pitch = dst.getPitch();
    p.dstPtr.xsize = dst.size[0];
    p.dstPtr.ysize = dst.size[1];
    p.extent.width = size[0];
    p.extent.height = size[1];
    p.extent.depth = size[2];
    p.kind = kind;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

/**
   Copy CUDA array to host memory.
   @param dst destination pointer (host memory)
   @param src source CUDA array
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const Array<Type, Dim> &src)
{
  copyFromArray(dst, src, cudaMemcpyDeviceToHost);
}

/**
   Copy region from CUDA array to host memory.
   @param dst destination pointer (host memory)
   @param src source CUDA array
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const Array<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)

{
  copyFromArray(dst, src, dst_ofs, src_ofs, size, cudaMemcpyDeviceToHost);
}

/**
   Copy CUDA array to device memory.
   @param dst destination pointer (device memory)
   @param src source CUDA array
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, const Array<Type, Dim> &src)
{
  copyFromArray(dst, src, cudaMemcpyDeviceToDevice);
}

/**
   Copy region from CUDA array to device memory.
   @param dst destination pointer (device memory)
   @param src source CUDA array
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, const Array<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  copyFromArray(dst, src, dst_ofs, src_ofs, size, cudaMemcpyDeviceToDevice);
}

//------------------------------------------------------------------------------
/**
   Generic copy method from host/device memory to CUDA array.
   It is not recommended to call this function directly since its correct
   behaviour depends on the kind parameter (just as the underlying CUDA
   functions).
   @param dst destination CUDA array
   @param src generic source pointer
   @param kind direction of copy
*/
template<class Type, unsigned Dim>
void
copyToArray(Array<Type, Dim> &dst, const Pointer<Type, Dim> &src, cudaMemcpyKind kind)
{
  CUDA_STATIC_ASSERT(Dim >= 2);
  CUDA_STATIC_ASSERT(Dim <= 3);
  CUDA_CHECK_SIZE;

  if(Dim == 2) {
    CUDA_CHECK(cudaMemcpy2DToArray(dst.getArray(), 0, 0,
				   src.getBuffer(), src.getPitch(),
				   src.size[0] * sizeof(Type), src.size[1], kind));
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcPtr.ptr = (void *)src.getBuffer();
    p.srcPtr.pitch = src.getPitch();
    p.srcPtr.xsize = src.size[0];
    p.srcPtr.ysize = src.size[1];
    p.dstArray = dst.getArray();
    p.dstPtr.pitch = dst.getPitch();
    p.dstPtr.xsize = dst.size[0];
    p.dstPtr.ysize = dst.size[1];
    p.extent.width = src.size[0];
    p.extent.height = src.size[1];
    p.extent.depth = src.size[2];

    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= src.size[i];

    p.kind = kind;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

/**
   Generic copy method from host/device memory to CUDA array.
   It is not recommended to call this function directly since its correct
   behaviour depends on the kind parameter (just as the underlying CUDA
   functions).
   @param dst destination CUDA array
   @param src generic source pointer
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
   @param kind direction of copy
*/
template<class Type, unsigned Dim>
void
copyToArray(Array<Type, Dim> &dst, const Pointer<Type, Dim> &src,
	    const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size,
	    cudaMemcpyKind kind)

{
  CUDA_STATIC_ASSERT(Dim >= 2);
  CUDA_STATIC_ASSERT(Dim <= 3);

  check_bounds(dst, src, dst_ofs, src_ofs, size);

  if(Dim == 2) {
    /*
      Notes:
      -) the dstX parameter of cudaMemcpy2DToArray is measured in bytes
      -) src.getBuffer() is a "Type *", i.e., we can count in elements here
    */
    CUDA_CHECK(cudaMemcpy2DToArray(dst.getArray(), dst_ofs[0] * sizeof(Type), dst_ofs[1],
				   src.getBuffer() + src_ofs[0] + src_ofs[1] * src.stride[0], src.getPitch(),
				   size[0] * sizeof(Type), size[1], kind));
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
#if CUDA_USE_OFFSET
    p.srcPos.x = src_ofs[0] * sizeof(Type);
    p.srcPos.y = src_ofs[1];
    p.srcPos.z = src_ofs[2];
    p.srcPtr.ptr = (void *)src.getBuffer();
#else
    p.srcPtr.ptr = (void *)(src.getBuffer() + src_ofs[0] + src_ofs[1] * src.stride[0] + src_ofs[2] * src.stride[1]);
#endif
    p.srcPtr.pitch = src.getPitch();
    p.srcPtr.xsize = src.size[0];
    p.srcPtr.ysize = src.size[1];
    p.dstArray = dst.getArray();
    p.dstPos.x = dst_ofs[0];
    p.dstPos.y = dst_ofs[1];
    p.dstPos.z = dst_ofs[2];
    p.dstPtr.pitch = dst.getPitch();
    p.dstPtr.xsize = dst.size[0];
    p.dstPtr.ysize = dst.size[1];
    p.extent.width = size[0];
    p.extent.height = size[1];
    p.extent.depth = size[2];

    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= src.size[i];

    p.kind = kind;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

/**
   Copy host memory to CUDA array.
   @param dst destination CUDA array
   @param src source pointer (host memory)
*/
template<class Type, unsigned Dim>
void
copy(Array<Type, Dim> &dst, const HostMemory<Type, Dim> &src)
{
  copyToArray(dst, src, cudaMemcpyHostToDevice);
}

/**
   Copy region from host memory to CUDA array.
   @param dst destination CUDA array
   @param src source pointer (host memory)
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(Array<Type, Dim> &dst, const HostMemory<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  copyToArray(dst, src, dst_ofs, src_ofs, size, cudaMemcpyHostToDevice);
}

/**
   Copy device memory to CUDA array.
   @param dst destination CUDA array
   @param src source pointer (device memory)
*/
template<class Type, unsigned Dim>
void
copy(Array<Type, Dim> &dst, const DeviceMemory<Type, Dim> &src)
{
  copyToArray(dst, src, cudaMemcpyDeviceToDevice);
}

/**
   Copy region from device memory to CUDA array.
   @param dst destination CUDA array
   @param src source pointer (device memory)
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(Array<Type, Dim> &dst, const DeviceMemory<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  copyToArray(dst, src, dst_ofs, src_ofs, size, cudaMemcpyDeviceToDevice);
}

/**
   Copy CUDA array to CUDA array.
   @param dst destination CUDA array
   @param src source CUDA array
*/
template<class Type, unsigned Dim>
void
copy(Array<Type, Dim> &dst, const Array<Type, Dim> &src)
{
  CUDA_STATIC_ASSERT(Dim >= 2);
  CUDA_STATIC_ASSERT(Dim <= 3);
  CUDA_CHECK_SIZE;

  if(Dim == 2) {
    CUDA_CHECK(cudaMemcpy2DArrayToArray(dst.getArray(), 0, 0,
					src.getArray(), 0, 0,
					src.size[0] * sizeof(Type), src.size[1], cudaMemcpyDeviceToDevice));
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcArray = const_cast<cudaArray *>(src.getArray());
    p.srcPtr.pitch = src.getPitch();
    p.srcPtr.xsize = src.size[0];
    p.srcPtr.ysize = src.size[1];
    p.dstArray = dst.getArray();
    p.dstPtr.pitch = dst.getPitch();
    p.dstPtr.xsize = dst.size[0];
    p.dstPtr.ysize = dst.size[1];
    p.extent.width = src.size[0];
    p.extent.height = src.size[1];
    p.extent.depth = src.size[2];

    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= src.size[i];

    p.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

/**
   Copy region from CUDA array to CUDA array.
   @param dst destination CUDA array
   @param src source CUDA array
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
*/
template<class Type, unsigned Dim>
void
copy(Array<Type, Dim> &dst, const Array<Type, Dim> &src,
     const Size<Dim> &dst_ofs, const Size<Dim> &src_ofs, const Size<Dim> &size)
{
  CUDA_STATIC_ASSERT(Dim >= 2);
  CUDA_STATIC_ASSERT(Dim <= 3);

  check_bounds(dst, src, dst_ofs, src_ofs, size);

  if(Dim == 2) {
    CUDA_CHECK(cudaMemcpy2DArrayToArray(dst.getArray(), dst_ofs[0] * sizeof(Type), dst_ofs[1],
					src.getArray(), src_ofs[0] * sizeof(Type), src_ofs[1],
					size[0] * sizeof(Type), size[1], cudaMemcpyDeviceToDevice));
  }
  else if(Dim == 3) {
    cudaMemcpy3DParms p = { 0 };
    p.srcArray = const_cast<cudaArray *>(src.getArray());
    p.srcPos.x = src_ofs[0];
    p.srcPos.y = src_ofs[1];
    p.srcPos.z = src_ofs[2];
    p.dstArray = dst.getArray();
    p.dstPos.x = dst_ofs[0];
    p.dstPos.y = dst_ofs[1];
    p.dstPos.z = dst_ofs[2];
    p.extent.width = size[0];
    p.extent.height = size[1];
    p.extent.depth = size[2];

    for(unsigned i = 3; i < Dim; ++i)
      p.extent.depth *= src.size[i];

    p.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&p));
  }
}

//------------------------------------------------------------------------------
/**
   Generic copy method from CUDA symbol to host/device memory.
   It is not recommended to call this function directly since its correct
   behaviour depends on the kind parameter (just as the underlying CUDA
   functions).
   @param dst generic destination pointer
   @param src source CUDA symbol
   @param kind direction of copy
*/
template<class Type, unsigned Dim>
void
copyFromSymbol(Pointer<Type, Dim> &dst, const Symbol<Type, Dim> &src, cudaMemcpyKind kind)
{
  CUDA_CHECK_SIZE;
  CUDA_CHECK(cudaMemcpyFromSymbol(dst.getBuffer(), src.getSymbol(), dst.getBytes(), 0, kind));
}

/**
   Copy CUDA symbol to host memory.
   @param dst destination pointer (host memory)
   @param src source symbol (device memory)
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const Symbol<Type, Dim> &src)
{
  copyFromSymbol(dst, src, cudaMemcpyDeviceToHost);
}

/**
   Copy CUDA symbol to device memory.
   @param dst destination pointer (device memory)
   @param src source symbol (device memory)
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, const Symbol<Type, Dim> &src)
{
  copyFromSymbol(dst, src, cudaMemcpyDeviceToDevice);
}

//------------------------------------------------------------------------------
/**
   Generic copy method from host/device memory to CUDA symbol.
   It is not recommended to call this function directly since its correct
   behaviour depends on the kind parameter (just as the underlying CUDA
   functions).
   @param dst destination CUDA symbol
   @param src generic source pointer
   @param kind direction of copy
*/
template<class Type, unsigned Dim>
void
copyToSymbol(Symbol<Type, Dim> &dst, const Pointer<Type, Dim> &src, cudaMemcpyKind kind)
{
  CUDA_CHECK_SIZE;
  CUDA_CHECK(cudaMemcpyToSymbol(dst.getSymbol(), src.getBuffer(), src.getBytes(), 0, kind));
}

/**
   Copy host memory to CUDA symbol.
   @param dst destination symbol (device memory)
   @param src source pointer (host memory)
*/
template<class Type, unsigned Dim>
void
copy(Symbol<Type, Dim> &dst, const HostMemory<Type, Dim> &src)
{
  copyToSymbol(dst, src, cudaMemcpyHostToDevice);
}

/**
   Copy device memory to CUDA symbol.
   @param dst destination symbol (device memory)
   @param src source pointer (device memory)
*/
template<class Type, unsigned Dim>
void
copy(Symbol<Type, Dim> &dst, const DeviceMemory<Type, Dim> &src)
{
  copyToSymbol(dst, src, cudaMemcpyDeviceToDevice);
}

//------------------------------------------------------------------------------
/**
   Generic copy method with border handling.
   @param dst destination
   @param src source
   @param dst_ofs destination offset
   @param src_ofs source offset
   @param size size of region to be copied
   @param border border handling for source data
*/
template<class TypeDst, class TypeSrc>
void
copy(TypeDst &dst, const TypeSrc &src,
     const Size<TypeDst::Dim> &dst_ofs, const SSize<TypeSrc::Dim> &src_ofs, const Size<TypeDst::Dim> &size,
     border_t border)
{
  CUDA_STATIC_ASSERT((unsigned)(TypeDst::Dim) == (unsigned)(TypeSrc::Dim));
  enum { Dim = TypeDst::Dim };
  Size<TypeDst::Dim> dst_ofs2 = dst_ofs;
  Size<TypeSrc::Dim> src_ofs2 = src_ofs;
  Size<TypeDst::Dim> size2 = size;

  for(unsigned i = TypeSrc::Dim; i--;) {
    // check minimum:
    if(src_ofs[i] < 0) {
      if(src_ofs[i] + size[i] < 0)
	CUDA_ERROR("source region must not be empty");

      size2[i] += src_ofs[i];
      dst_ofs2[i] -= src_ofs[i];
      src_ofs2[i] = 0;
    }

    // check maximum:
    if(src_ofs[i] + size[i] >= src.size[i]) {
      if(src_ofs[i] >= src.size[i])
	CUDA_ERROR("source region must not be empty");

      size2[i] -= src_ofs[i] + size[i] - src.size[i];
    }
  }

  // copy interior data:
  copy(dst, src, dst_ofs2, src_ofs2, size2);

  // border handling:
  Size<Dim> dst_ofs3 = dst_ofs2;
  Size<Dim> src_ofs3 = dst_ofs2;
  Size<Dim> size3 = size2;

  for(unsigned i = Dim; i--;) {
    /*
    for(unsigned j = Dim; j--;) {
      if(j > i) {
	// dimension j has already been processed:
	src_ofs3[j] = dst_ofs3[k] = dst_ofs[k];
	size3[k] = size[k];
      }
      else if(j < i) {
	// dimension j has not yet been processed:
	src_ofs3[j] = dst_ofs3[k] = dst_ofs2[k];
	size3[k] = size2[k];
      }
      // the case j == i is handled below
    }
    */

    if(src_ofs[i] < 0) {
      for(unsigned j = -src_ofs[i]; j--;) {
	// find region to be copied according to border mode:
	switch(border) {
	case BORDER_CLAMP:
	  src_ofs3[i] = dst_ofs2[i];
	  break;

	default:
	  CUDA_ERROR("border mode not yet implemented");
	}

	dst_ofs3[i] = dst_ofs2[i] - 1 - j;
	size3[i] = 1;
	copy(dst, dst, dst_ofs3, src_ofs3, size3);
      }
    }

    if(src_ofs[i] + size[i] >= src.size[i]) {
      for(unsigned j = src_ofs[i] + size[i] - src.size[i]; j--;) {
	// find region to be copied according to border mode:
	switch(border) {
	case BORDER_CLAMP:
	  src_ofs3[i] = dst_ofs2[i] + size2[i] - 1;
	  break;

	default:
	  CUDA_ERROR("border mode not yet implemented");
	}

	dst_ofs3[i] = dst_ofs2[i] + size2[i] + j;
	size3[i] = 1;
	copy(dst, dst, dst_ofs3, src_ofs3, size3);
      }
    }

    // processing of dimension i is complete,
    // use entire extent in dimension i for processing of remaining dimensions:
    src_ofs3[i] = dst_ofs3[i] = dst_ofs[i];
    size3[i] = size[i];
  }
}

//------------------------------------------------------------------------------
/**
   Copy constant value to region in host memory.
   @param dst destination pointer (host memory)
   @param val value to copy
   @param dst_ofs destination offset
   @param size size of region
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const Type &val,
     const Size<Dim> &dst_ofs, const Size<Dim> &size)
{
  dst.checkBounds(dst_ofs, size);
  typename HostMemory<Type, Dim>::iterator i(dst_ofs, Size<Dim>(dst_ofs + size)), iend(i);
  iend.setEnd();

  for(; i != iend; ++i)
    dst[i] = val;
}

/**
   Copy constant value to host memory.
   @param dst destination pointer (host memory)
   @param val value to copy
*/
template<class Type, unsigned Dim>
void
copy(HostMemory<Type, Dim> &dst, const Type &val)
{
  copy(dst, val, Size<Dim>(), dst.size);
}

#if defined(__CUDACC__) || defined(__DOXYGEN__)

/**
   Dummy class for kernel instantiation.
   nvcc gets confused if it doesn't see at least one instatiation of template
   kernels. The constructor of this dummy class provides one.
*/
struct DummyInstantiateCopyConstantKernels
{
  DummyInstantiateCopyConstantKernels()
  {
    dim3 gridDim(1, 1, 1), blockDim(1, 1, 1);
    Size<1> r1;
    int val = 0;
    typename DeviceMemory<int, 1>::KernelData kdst;
    copy_constant_nocheck_kernel<<<gridDim, blockDim>>>(kdst, val, Dimension<1>());
    copy_constant_check_kernel<<<gridDim, blockDim>>>(kdst, val, r1, r1);
  }
};

/**
   Copy constant value to region in device memory.
   Since this function calls a CUDA kernel, it is only available if the file
   from which this function is called is compiled by nvcc.
   @param dst destination pointer (device memory)
   @param val value to copy
   @param dst_ofs destination offset
   @param size size of region
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, Type val,
     const Size<Dim> &dst_ofs, const Size<Dim> &size)
{
  dst.checkBounds(dst_ofs, size);
  dim3 gridDim, blockDim;
  bool aligned;
  size_t dofs;
  Size<Dim> rmin, rmax;
  dst.getExecutionConfiguration(gridDim, blockDim, aligned, dofs, rmin, rmax, dst_ofs, size);
  typename DeviceMemory<Type, Dim>::KernelData kdst(dst);
  kdst.data += dofs;

  if(aligned)
    copy_constant_nocheck_kernel<<<gridDim, blockDim>>>(kdst, val, Dimension<Dim>());
  else
    copy_constant_check_kernel<<<gridDim, blockDim>>>(kdst, val, rmin, rmax);

  CUDA_CHECK(cudaGetLastError());
}

/**
   Copy constant value to device memory.
   Since this function calls a CUDA kernel, it is only available if the file
   from which this function is called is compiled by nvcc.
   @param dst destination pointer (device memory)
   @param val value to copy
*/
template<class Type, unsigned Dim>
void
copy(DeviceMemory<Type, Dim> &dst, Type val)
{
  copy(dst, val, Size<Dim>(), dst.size);
}

#endif  // defined(__CUDACC__) || defined(__DOXYGEN__)

}  // namespace Cuda


#endif
