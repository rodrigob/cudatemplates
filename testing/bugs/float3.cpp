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
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>

using namespace std;


int
main()
{
  std::cout << "1" << std::endl;
  Cuda::DeviceMemoryLinear<float3,2> test1(Cuda::Size<2>(584,388));
  std::cout << "2" << std::endl;
  Cuda::DeviceMemoryPitched<float4,2> test2(Cuda::Size<2>(584,388));
  std::cout << "3" << std::endl;
  Cuda::DeviceMemoryPitched<float3,2> test31(Cuda::Size<2>(123,456));
  std::cout << "4" << std::endl;
  Cuda::DeviceMemoryPitched<float3,2> test32(Cuda::Size<2>(456,123)); // problem
  std::cout << "5" << std::endl;
  Cuda::DeviceMemoryPitched<float3,2> test4(Cuda::Size<2>(584,388)); // problem
  std::cout << "6" << std::endl;
  return 0;
}
