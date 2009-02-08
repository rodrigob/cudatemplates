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

#include <iostream>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/hostmemoryheap.hpp>

using namespace std;


void
copy_border(Cuda::HostMemoryHeap2D<int> &dst, Cuda::HostMemoryHeap2D<int> &src, Cuda::SSize<2> pos)
{
  cout << endl << pos[0] << ' ' << pos[1] << ":\n";

  pos[0] -= 1;
  pos[1] -= 1;
  copy(dst, src, Cuda::Size<2>(0, 0), pos, Cuda::Size<2>(3, 3), Cuda::BORDER_CLAMP);

  for(int y = 0; y < 3; ++y) {
    for(int x = 0; x < 3; ++x)
      cout << dst[Cuda::Size<2>(x, y)] << ' ';

    cout << endl;
  }
}

int
main()
{
  const int width = 5, height = 5;
  Cuda::Size<2> size(width, height);
  Cuda::HostMemoryHeap2D<int> src(size);
  Cuda::HostMemoryHeap2D<int> dst(Cuda::Size<2>(3, 3));

  for(int y = height; y--;)
    for(int x = width; x--;)
      src[Cuda::Size<2>(x, y)] = x + y;
  
  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x)
      cout << src[Cuda::Size<2>(x, y)] << ' ';

    cout << endl;
  }

  copy_border(dst, src, Cuda::SSize<2>(2, 2));
  copy_border(dst, src, Cuda::SSize<2>(0, 0));
  copy_border(dst, src, Cuda::SSize<2>(1, 0));
  copy_border(dst, src, Cuda::SSize<2>(3, 0));
  copy_border(dst, src, Cuda::SSize<2>(4, 0));
  copy_border(dst, src, Cuda::SSize<2>(0, 1));
  copy_border(dst, src, Cuda::SSize<2>(0, 3));
  copy_border(dst, src, Cuda::SSize<2>(0, 4));
  copy_border(dst, src, Cuda::SSize<2>(1, 4));
  copy_border(dst, src, Cuda::SSize<2>(3, 4));
  copy_border(dst, src, Cuda::SSize<2>(4, 4));
  copy_border(dst, src, Cuda::SSize<2>(4, 3));
  copy_border(dst, src, Cuda::SSize<2>(4, 1));
}
