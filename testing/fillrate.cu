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

#include <typeinfo>

#include <stdio.h>

#include <cudatemplates/copy_constant.hpp>
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/event.hpp>


/**
   Compute gigabytes per second.
   @param size edge length of square (i.e., size*size pixels in total)
   @param ms time in milliseconds
*/
template <class T>
float
gbps(int size, float ms)
{
  int bytes = size * size * sizeof(T);
  return bytes * 1000.0 / ms / (1 << 30);
}

template <class T>
void fillrate()
{
  printf("fill rate for type '%s':\n", typeid(T).name());

  for(int i = 1; i <= 1024; ++i) {
    Cuda::Event event0, event1, event2;
    Cuda::DeviceMemoryPitched2D<T> image(i, i);

    int count = (1 << 12) / i;
    event0.record();

    // measure initMem method:
    for(int j = count; j--;)
      image.initMem(0, false);

    event1.record();

    // measure Cuda::copy method:
    for(int j = count; j--;)
      Cuda::copy(image, (T)0);

    event2.record();

    // print result:
    event2.synchronize();
    float t1 = (event1 - event0) / count;
    float t2 = (event2 - event1) / count;
    float gbps1 = gbps<T>(i, t1);
    float gbps2 = gbps<T>(i, t2);
    printf("%4d x %4d: initMem %f ms (%f GB/sec), Cuda::copy %f ms (%f GB/sec)\n", i, i, t1, gbps1, t2, gbps2);
  }
}

int
main()
{
  fillrate<char>();
  fillrate<short>();
  fillrate<int>();
  fillrate<float>();
}
