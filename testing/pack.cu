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

#include <typeinfo>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/event.hpp>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/pack.hpp>


const size_t SIZE = 1024;
const int COUNT = 100;


float
frand()
{
  return rand() / (float)RAND_MAX;
}

template <class VectorType, unsigned DataDim>
float
gbps(float ms)
{
  int bytes = sizeof(VectorType) * 2;  // one read plus one write transfer of vector size

  for(int i = DataDim; i--;)
    bytes *= SIZE;

  float gb = bytes / (float)(1 << 30);
  float sec = ms / (1000 * COUNT);
  return gb / sec;
}

template <class ScalarType, unsigned VectorDim, class VectorType, unsigned DataDim>
void
test()
{
  Cuda::Size<DataDim> size;

  for(int i = DataDim; i--;)
    size[i] = SIZE;

  Cuda::HostMemoryHeap<ScalarType, DataDim> h_data_scalar1[VectorDim], h_data_scalar2[VectorDim];
  Cuda::HostMemoryHeap<VectorType, DataDim> h_data_vector(size);
  Cuda::DeviceMemoryLinear<ScalarType, DataDim> d_data_scalar1[VectorDim], d_data_scalar2[VectorDim];
  Cuda::DeviceMemoryLinear<VectorType, DataDim> d_data_vector(size);

  for(int i = VectorDim; i--;) {
    // allocate host memory:
    h_data_scalar1[i].alloc(size);
    h_data_scalar2[i].alloc(size);

    // allocate device memory:
    d_data_scalar1[i].alloc(size);
    d_data_scalar2[i].alloc(size);

    // initialize data:
    for(Cuda::Iterator<DataDim> j = h_data_scalar1[i].begin(); j != h_data_scalar1[i].end(); ++j)
      h_data_scalar1[i][j] = frand();

    // copy data from host to device memory:
    Cuda::copy(d_data_scalar1[i], h_data_scalar1[i]);
  }

  Cuda::Event t0, t1, t2;

  // pack scalars into vector:
  t0.record();
  
  for(int i = COUNT; i--;)
    Cuda::pack(d_data_vector, d_data_scalar1[0], d_data_scalar1[1], d_data_scalar1[2], d_data_scalar1[3]);

  // unpack vector into scalars:
  t1.record();
  
  for(int i = COUNT; i--;)
    Cuda::unpack(d_data_scalar2[0], d_data_scalar2[1], d_data_scalar2[2], d_data_scalar2[3], d_data_vector);

  // report performance:
  t2.record();
  t2.synchronize();
  printf("pack   %dD %s: %f GB / sec\n", VectorDim, typeid(ScalarType).name(), gbps<VectorType, DataDim>(t1 - t0));
  printf("unpack %dD %s: %f GB / sec\n", VectorDim, typeid(ScalarType).name(), gbps<VectorType, DataDim>(t2 - t1));

  // copy data from device to host memory:
  Cuda::copy(h_data_vector, d_data_vector);

  for(int i = VectorDim; i--;)
    Cuda::copy(h_data_scalar2[i], d_data_scalar2[i]);

  // verify packed data:
  for(Cuda::Iterator<DataDim> i = h_data_vector.begin(); i != h_data_vector.end(); ++i) {
    const ScalarType *vec = &h_data_vector[i].x;

    for(int j = VectorDim; j--;) {
      assert(h_data_scalar2[j][i] == h_data_scalar1[j][i]);
      assert(vec[j] == h_data_scalar2[j][i]);
    }
  }
}

int
main()
{
  test<unsigned char, 4, uchar4, 2>();
  test<short, 4, short4, 2>();
  test<int, 4, int4, 2>();
  test<float, 4, float4, 2>();

  return 0;
}
