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

#ifndef CUDA_DIMENSION_H
#define CUDA_DIMENSION_H


namespace Cuda {

/*
  We need template kernels for handling different copy and conversion cases.
  Since nvcc can't process template kernels with typename arguments (e.g.,
  ...::KernelData), and partial function template specializations are not yet
  supported (see http://www.gotw.ca/publications/mill17.htm for more
  information), we use the dummy template class "Cuda::Dimension" to pass the
  dimension information (BTW, removing the "dummy" arguments from the kernel
  signatures results in a segfault in cudafe).
*/
template <unsigned N>
struct Dimension
{
};

}  // namespace Cuda


#endif
