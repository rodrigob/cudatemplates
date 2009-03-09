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
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>

using namespace std;


int
main()
{
  Cuda::Size<1> size(26);
  Cuda::HostMemoryHeap1D<char> h_data1(size), h_data2(size);
  Cuda::DeviceMemoryLinear1D<char> d_data(size);

  for(unsigned i = 26; i--;)
    h_data1[i] = 'a' + i;

  h_data1[26] = 0;
  Cuda::copy(d_data, h_data1);
  cudaMemset(d_data.getBuffer(), '-', 13);
  Cuda::copy(h_data2, d_data);
  cout << h_data2.getBuffer() << endl;
  return 0;
}
