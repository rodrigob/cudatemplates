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
#include <stdlib.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/pack.hpp>


float
frand()
{
  return rand() / (float)RAND_MAX;
}

int
main()
{
  const size_t SIZE = 64;
  const Cuda::Size<2> size(SIZE, SIZE);

  Cuda::HostMemoryHeap<float , 2> h_data1x(size), h_data1y(size), h_data1z(size), h_data1w(size);
  Cuda::HostMemoryHeap<float2, 2> h_data2(size);
  Cuda::HostMemoryHeap<float3, 2> h_data3(size);
  Cuda::HostMemoryHeap<float4, 2> h_data4(size);

  Cuda::DeviceMemoryLinear<float , 2> d_data1x(size), d_data1y(size), d_data1z(size), d_data1w(size);
  Cuda::DeviceMemoryLinear<float2, 2> d_data2(size);
  Cuda::DeviceMemoryLinear<float3, 2> d_data3(size);
  Cuda::DeviceMemoryLinear<float4, 2> d_data4(size);

  for(Cuda::Iterator<2> i = h_data1x.begin(); i != h_data1x.end(); ++i) {
    h_data1x[i] = frand();
    h_data1y[i] = frand();
    h_data1z[i] = frand();
    h_data1w[i] = frand();
  }      

  Cuda::copy(d_data1x, h_data1x);
  Cuda::copy(d_data1y, h_data1y);
  Cuda::copy(d_data1z, h_data1z);
  Cuda::copy(d_data1w, h_data1w);

  Cuda::pack(d_data2, d_data1x, d_data1y);
  Cuda::pack(d_data3, d_data1x, d_data1y, d_data1z);
  Cuda::pack(d_data4, d_data1x, d_data1y, d_data1z, d_data1w);

  Cuda::copy(h_data2, d_data2);
  Cuda::copy(h_data3, d_data3);
  Cuda::copy(h_data4, d_data4);

  for(Cuda::Iterator<2> i = h_data1x.begin(); i != h_data1x.end(); ++i) {
    assert(h_data2[i].x == h_data1x[i]);
    assert(h_data2[i].y == h_data1y[i]);

    assert(h_data3[i].x == h_data1x[i]);
    assert(h_data3[i].y == h_data1y[i]);
    assert(h_data3[i].z == h_data1z[i]);

    assert(h_data4[i].x == h_data1x[i]);
    assert(h_data4[i].y == h_data1y[i]);
    assert(h_data4[i].z == h_data1z[i]);
    assert(h_data4[i].w == h_data1w[i]);
  }      

  return 0;
}
