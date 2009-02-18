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

#include <cudatemplates/copy.hpp>
#include <cudatemplates/copy_constant.hpp>
#include <cudatemplates/devicememory.hpp>

/*
  Explicit template instantiations.
  Unless you can put all calls to functions defined in copy_constant.hpp in a
  single file, you need to explicitly instantiate the template functions to
  avoid "multiply defined symbols" errors in the linker. See below for an
  example.
*/
namespace Cuda {
  template void copy<float, 1>(DeviceMemory<float, 1>&, float);
  template void copy<float, 1>(DeviceMemory<float, 1>&, float, Size<1> const&, Size<1> const&);
  template void copy<float, 2>(DeviceMemory<float, 2>&, float);
  template void copy<float, 2>(DeviceMemory<float, 2>&, float, Size<2> const&, Size<2> const&);
  template void copy<int, 1>(DeviceMemory<int, 1>&, int);
  template void copy<int, 1>(DeviceMemory<int, 1>&, int, Size<1> const&, Size<1> const&);
  template void copy<int, 2>(DeviceMemory<int, 2>&, int);
  template void copy<int, 2>(DeviceMemory<int, 2>&, int, Size<2> const&, Size<2> const&);
}
