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
#include <cudatemplates/convert.hpp>
#include <cudatemplates/devicememorylinear.hpp>


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
