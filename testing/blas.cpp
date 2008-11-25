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

#include <sys/time.h>

#include <cstdlib>
#include <iostream>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/cublas.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>

using namespace std;


const int   SIZE    =   1024;  // image size
const int   COUNT   = 100000;  // number of FFTs to perform
const float EPSILON =   1e-6;  // error threshold


double
operator-(const struct timeval &t1, const struct timeval &t2)
{
  return (t1.tv_sec - t2.tv_sec) + (t1.tv_usec - t2.tv_usec) * 1e-6;
}

int
main()
{
  // init CUBLAS:
  Cuda::BLAS::init();

  // create single data:
  float sa = 3;
  Cuda::BLAS::Vector<float> sx(3), sy(3);
  Cuda::BLAS::Matrix<float> sm(3, 3);
  Cuda::BLAS::complex ca;

  // create complex data:
  ca.x = sa;
  ca.y = 0;
  Cuda::BLAS::Vector<Cuda::BLAS::complex> cx(3), cy(3);
  Cuda::BLAS::Matrix<Cuda::BLAS::complex> cm(3, 3);

  // apply single functions:
  Cuda::BLAS::axpy(sa, sx, sy);
  Cuda::BLAS::gemv('n', sa, sm, sx, sa, sy);

  // apply complex functions:
  Cuda::BLAS::axpy(ca, cx, cy);
  // Cuda::BLAS::gemv(ca, cm, cx, ca, cy);  // not yet implemented by NVIDIA

  return 0;
}
