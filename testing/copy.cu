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

#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/hostmemoryheap.hpp>


extern int err;


template <class T>
int
test_array_init1(size_t size_max)
{
  // random object size:
  Cuda::Size<T::Dim> size0, ofs, size;

  for(size_t i = T::Dim; i--;) {
    size0[i] = (rand() % (size_max - 1)) + 1;
    ofs[i] = (size0[i] > 1) ? rand() % (size0[i] - 1) : 0;
    size_t r = size0[i] - ofs[i];
    size[i] = (r > 1) ? (rand() % (r - 1)) + 1 : 1;
  }

  // random values:
  typename T::Type val1, val2;
  typename T::Type div = 1 << 16;
  val1 = rand() / div;
  val2 = rand() / div;

  // create object and set values:
  T data(size0);
  copy(data, val1);
  copy(data, val2, ofs, size);

  // verify data:
  Cuda::HostMemoryHeap<typename T::Type, T::Dim> hdata(data);
  
  for(Cuda::Iterator<T::Dim> index = hdata.begin(); index != hdata.end(); ++index) {
    bool inside = true;

    for(size_t i = T::Dim; i--;)
      if((index[i] < ofs[i]) || (index[i] >= ofs[i] + size[i])) {
	inside = false;
	break;
      }

    if(hdata[index] != (inside ? val2 : val1)) {
      fprintf(stderr, "array init test failed\n");
      return 1;
    }
  }

  return 0;
}

void
test_array_init()
{
  err |= test_array_init1<Cuda::HostMemoryHeap<float, 1> >(1 << 15);
  err |= test_array_init1<Cuda::HostMemoryHeap<int  , 1> >(1 << 15);
  err |= test_array_init1<Cuda::HostMemoryHeap<float, 2> >(1 << 10);
  err |= test_array_init1<Cuda::HostMemoryHeap<int  , 2> >(1 << 10);
  err |= test_array_init1<Cuda::HostMemoryHeap<float, 3> >(1 <<  5);
  err |= test_array_init1<Cuda::HostMemoryHeap<int  , 3> >(1 <<  5);

  err |= test_array_init1<Cuda::DeviceMemoryLinear<float, 1> >(1 << 15);
  err |= test_array_init1<Cuda::DeviceMemoryLinear<int  , 1> >(1 << 15);
  err |= test_array_init1<Cuda::DeviceMemoryLinear<float, 2> >(1 << 10);
  err |= test_array_init1<Cuda::DeviceMemoryLinear<int  , 2> >(1 << 10);

  err |= test_array_init1<Cuda::DeviceMemoryPitched<float, 2> >(1 << 10);
  err |= test_array_init1<Cuda::DeviceMemoryPitched<int  , 2> >(1 << 10);

  // init 3D data in device memory not yet supported
}
