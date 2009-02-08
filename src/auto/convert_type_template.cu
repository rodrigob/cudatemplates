

// conversion from @type2@ to @type1@:

__global__ void convert_type_nocheck_@type1_@_@type2_@_@dim@_kernel
  (Cuda::DeviceMemory<@type1@, @dim@>::KernelData dst,
   Cuda::DeviceMemory<@type2@, @dim@>::KernelData src)
{@coords@
  dst.data[@ofs_dst@] = src.data[@ofs_src@];
}

__global__ void convert_type_check_@type1_@_@type2_@_@dim@_kernel
  (Cuda::DeviceMemory<@type1@, @dim@>::KernelData dst,
   Cuda::DeviceMemory<@type2@, @dim@>::KernelData src)
{@coords@

  if(@cond@)
    dst.data[@ofs_dst@] = src.data[@ofs_src@];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<@type1@, @dim@>::KernelData &dst,
   DeviceMemory<@type2@, @dim@>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_@type1_@_@type2_@_@dim@_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<@type1@, @dim@>::KernelData &dst,
   DeviceMemory<@type2@, @dim@>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_@type1_@_@type2_@_@dim@_kernel<<<gridDim, blockDim>>>(dst, src);
}

}
