/*
  NOTE: THIS FILE HAS BEEN CREATED AUTOMATICALLY,
  ANY CHANGES WILL BE OVERWRITTEN WITHOUT NOTICE!
*/

/* 
  Cuda Templates.

  Copyright (C) 2008 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  
  This program is free software you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <cudatemplates/convert.hpp>
#include <cudatemplates/devicememory.hpp>


// conversion from unsigned char to char:

__global__ void convert_type_nocheck_char_unsigned_char_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_char_unsigned_char_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to char:

__global__ void convert_type_nocheck_char_short_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_char_short_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to char:

__global__ void convert_type_nocheck_char_unsigned_short_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_char_unsigned_short_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to char:

__global__ void convert_type_nocheck_char_int_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_char_int_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to char:

__global__ void convert_type_nocheck_char_unsigned_int_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_char_unsigned_int_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to char:

__global__ void convert_type_nocheck_char_float_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_char_float_1_kernel
  (Cuda::DeviceMemory<char, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_char_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_char_char_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_short_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_char_short_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_unsigned_short_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_char_unsigned_short_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_int_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_char_int_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_unsigned_int_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_char_unsigned_int_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_float_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_char_float_1_kernel
  (Cuda::DeviceMemory<unsigned char, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to short:

__global__ void convert_type_nocheck_short_char_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_short_char_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to short:

__global__ void convert_type_nocheck_short_unsigned_char_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_short_unsigned_char_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to short:

__global__ void convert_type_nocheck_short_unsigned_short_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_short_unsigned_short_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to short:

__global__ void convert_type_nocheck_short_int_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_short_int_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to short:

__global__ void convert_type_nocheck_short_unsigned_int_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_short_unsigned_int_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to short:

__global__ void convert_type_nocheck_short_float_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_short_float_1_kernel
  (Cuda::DeviceMemory<short, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_char_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_short_char_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_unsigned_char_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_short_unsigned_char_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_short_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_short_short_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_int_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_short_int_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_unsigned_int_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_short_unsigned_int_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_float_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_short_float_1_kernel
  (Cuda::DeviceMemory<unsigned short, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to int:

__global__ void convert_type_nocheck_int_char_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_int_char_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to int:

__global__ void convert_type_nocheck_int_unsigned_char_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_int_unsigned_char_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to int:

__global__ void convert_type_nocheck_int_short_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_int_short_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to int:

__global__ void convert_type_nocheck_int_unsigned_short_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_int_unsigned_short_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to int:

__global__ void convert_type_nocheck_int_unsigned_int_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_int_unsigned_int_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to int:

__global__ void convert_type_nocheck_int_float_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_int_float_1_kernel
  (Cuda::DeviceMemory<int, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_char_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_int_char_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_unsigned_char_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_int_unsigned_char_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_short_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_int_short_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_unsigned_short_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_int_unsigned_short_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_int_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_int_int_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_float_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_unsigned_int_float_1_kernel
  (Cuda::DeviceMemory<unsigned int, 1>::KernelData dst,
   Cuda::DeviceMemory<float, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 1>::KernelData &dst,
   DeviceMemory<float, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_float_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to float:

__global__ void convert_type_nocheck_float_char_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_float_char_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to float:

__global__ void convert_type_nocheck_float_unsigned_char_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_float_unsigned_char_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<unsigned char, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_char_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to float:

__global__ void convert_type_nocheck_float_short_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_float_short_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to float:

__global__ void convert_type_nocheck_float_unsigned_short_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_float_unsigned_short_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<unsigned short, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_short_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to float:

__global__ void convert_type_nocheck_float_int_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_float_int_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to float:

__global__ void convert_type_nocheck_float_unsigned_int_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  dst.data[x] = src.data[x];
}

__global__ void convert_type_check_float_unsigned_int_1_kernel
  (Cuda::DeviceMemory<float, 1>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 1>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;

  if(x < dst.size[0])
    dst.data[x] = src.data[x];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 1>::KernelData &dst,
   DeviceMemory<unsigned int, 1>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_int_1_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to char:

__global__ void convert_type_nocheck_char_unsigned_char_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_char_unsigned_char_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to char:

__global__ void convert_type_nocheck_char_short_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_char_short_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to char:

__global__ void convert_type_nocheck_char_unsigned_short_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_char_unsigned_short_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to char:

__global__ void convert_type_nocheck_char_int_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_char_int_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to char:

__global__ void convert_type_nocheck_char_unsigned_int_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_char_unsigned_int_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to char:

__global__ void convert_type_nocheck_char_float_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_char_float_2_kernel
  (Cuda::DeviceMemory<char, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_char_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_char_char_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_short_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_char_short_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_unsigned_short_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_char_unsigned_short_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_int_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_char_int_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_unsigned_int_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_char_unsigned_int_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_float_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_char_float_2_kernel
  (Cuda::DeviceMemory<unsigned char, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to short:

__global__ void convert_type_nocheck_short_char_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_short_char_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to short:

__global__ void convert_type_nocheck_short_unsigned_char_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_short_unsigned_char_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to short:

__global__ void convert_type_nocheck_short_unsigned_short_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_short_unsigned_short_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to short:

__global__ void convert_type_nocheck_short_int_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_short_int_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to short:

__global__ void convert_type_nocheck_short_unsigned_int_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_short_unsigned_int_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to short:

__global__ void convert_type_nocheck_short_float_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_short_float_2_kernel
  (Cuda::DeviceMemory<short, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_char_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_short_char_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_unsigned_char_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_short_unsigned_char_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_short_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_short_short_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_int_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_short_int_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_unsigned_int_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_short_unsigned_int_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_float_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_short_float_2_kernel
  (Cuda::DeviceMemory<unsigned short, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to int:

__global__ void convert_type_nocheck_int_char_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_int_char_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to int:

__global__ void convert_type_nocheck_int_unsigned_char_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_int_unsigned_char_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to int:

__global__ void convert_type_nocheck_int_short_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_int_short_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to int:

__global__ void convert_type_nocheck_int_unsigned_short_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_int_unsigned_short_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to int:

__global__ void convert_type_nocheck_int_unsigned_int_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_int_unsigned_int_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to int:

__global__ void convert_type_nocheck_int_float_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_int_float_2_kernel
  (Cuda::DeviceMemory<int, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
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


// conversion from char to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_char_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_int_char_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_unsigned_char_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_int_unsigned_char_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_short_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_int_short_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_unsigned_short_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_int_unsigned_short_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_int_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_int_int_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_float_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_unsigned_int_float_2_kernel
  (Cuda::DeviceMemory<unsigned int, 2>::KernelData dst,
   Cuda::DeviceMemory<float, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 2>::KernelData &dst,
   DeviceMemory<float, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_float_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to float:

__global__ void convert_type_nocheck_float_char_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_float_char_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to float:

__global__ void convert_type_nocheck_float_unsigned_char_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_float_unsigned_char_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<unsigned char, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_char_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to float:

__global__ void convert_type_nocheck_float_short_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_float_short_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to float:

__global__ void convert_type_nocheck_float_unsigned_short_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_float_unsigned_short_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<unsigned short, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_short_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to float:

__global__ void convert_type_nocheck_float_int_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_float_int_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to float:

__global__ void convert_type_nocheck_float_unsigned_int_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

__global__ void convert_type_check_float_unsigned_int_2_kernel
  (Cuda::DeviceMemory<float, 2>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 2>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x < dst.size[0] && y < dst.size[1])
    dst.data[x + y * dst.stride[0]] = src.data[x + y * src.stride[0]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 2>::KernelData &dst,
   DeviceMemory<unsigned int, 2>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_int_2_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to char:

__global__ void convert_type_nocheck_char_unsigned_char_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_char_unsigned_char_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to char:

__global__ void convert_type_nocheck_char_short_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_char_short_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to char:

__global__ void convert_type_nocheck_char_unsigned_short_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_char_unsigned_short_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to char:

__global__ void convert_type_nocheck_char_int_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_char_int_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to char:

__global__ void convert_type_nocheck_char_unsigned_int_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_char_unsigned_int_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to char:

__global__ void convert_type_nocheck_char_float_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_char_float_3_kernel
  (Cuda::DeviceMemory<char, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_char_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<char, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_char_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_char_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_char_char_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_short_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_char_short_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_unsigned_short_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_char_unsigned_short_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_int_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_char_int_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_unsigned_int_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_char_unsigned_int_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned char:

__global__ void convert_type_nocheck_unsigned_char_float_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_char_float_3_kernel
  (Cuda::DeviceMemory<unsigned char, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_char_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned char, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_char_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to short:

__global__ void convert_type_nocheck_short_char_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_short_char_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to short:

__global__ void convert_type_nocheck_short_unsigned_char_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_short_unsigned_char_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to short:

__global__ void convert_type_nocheck_short_unsigned_short_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_short_unsigned_short_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to short:

__global__ void convert_type_nocheck_short_int_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_short_int_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to short:

__global__ void convert_type_nocheck_short_unsigned_int_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_short_unsigned_int_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to short:

__global__ void convert_type_nocheck_short_float_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_short_float_3_kernel
  (Cuda::DeviceMemory<short, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_short_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<short, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_short_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_char_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_short_char_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_unsigned_char_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_short_unsigned_char_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_short_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_short_short_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_int_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_short_int_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_unsigned_int_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_short_unsigned_int_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned short:

__global__ void convert_type_nocheck_unsigned_short_float_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_short_float_3_kernel
  (Cuda::DeviceMemory<unsigned short, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_short_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned short, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_short_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to int:

__global__ void convert_type_nocheck_int_char_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_int_char_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to int:

__global__ void convert_type_nocheck_int_unsigned_char_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_int_unsigned_char_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to int:

__global__ void convert_type_nocheck_int_short_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_int_short_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to int:

__global__ void convert_type_nocheck_int_unsigned_short_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_int_unsigned_short_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to int:

__global__ void convert_type_nocheck_int_unsigned_int_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_int_unsigned_int_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to int:

__global__ void convert_type_nocheck_int_float_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_int_float_3_kernel
  (Cuda::DeviceMemory<int, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_int_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<int, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_int_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_char_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_int_char_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_unsigned_char_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_int_unsigned_char_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_short_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_int_short_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_unsigned_short_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_int_unsigned_short_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_int_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_int_int_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from float to unsigned int:

__global__ void convert_type_nocheck_unsigned_int_float_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_unsigned_int_float_3_kernel
  (Cuda::DeviceMemory<unsigned int, 3>::KernelData dst,
   Cuda::DeviceMemory<float, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_unsigned_int_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<unsigned int, 3>::KernelData &dst,
   DeviceMemory<float, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_unsigned_int_float_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from char to float:

__global__ void convert_type_nocheck_float_char_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_float_char_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned char to float:

__global__ void convert_type_nocheck_float_unsigned_char_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_float_unsigned_char_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned char, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<unsigned char, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_char_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from short to float:

__global__ void convert_type_nocheck_float_short_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_float_short_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned short to float:

__global__ void convert_type_nocheck_float_unsigned_short_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_float_unsigned_short_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned short, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<unsigned short, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_short_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from int to float:

__global__ void convert_type_nocheck_float_int_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_float_int_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}


// conversion from unsigned int to float:

__global__ void convert_type_nocheck_float_unsigned_int_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;
  dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

__global__ void convert_type_check_float_unsigned_int_3_kernel
  (Cuda::DeviceMemory<float, 3>::KernelData dst,
   Cuda::DeviceMemory<unsigned int, 3>::KernelData src)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if(x < dst.size[0] && y < dst.size[1] && z < dst.size[2])
    dst.data[x + y * dst.stride[0] + z * dst.stride[1]] = src.data[x + y * src.stride[0] + z * src.stride[1]];
}

namespace Cuda {

template <> void convert_type_nocheck
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_nocheck_float_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

template <> void convert_type_check
  (DeviceMemory<float, 3>::KernelData &dst,
   DeviceMemory<unsigned int, 3>::KernelData &src,
   dim3 gridDim, dim3 blockDim)
{
  convert_type_check_float_unsigned_int_3_kernel<<<gridDim, blockDim>>>(dst, src);
}

}
