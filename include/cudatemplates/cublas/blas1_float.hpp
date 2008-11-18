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


/*
  Note to cudatemplates developers:
  don't modify this file directly, modify the corresponding "*.hpp.in" file instead and re-run cmake!
*/


template <int N>
int iamax(const Vector<float, N> &x)
{
  return cublasIsamax(N, x, x.inc());
}

template <int N>
int iamin(const Vector<float, N> &x)
{
  return cublasIsamin(N, x, x.inc());
}

template <int N>
float asum(const Vector<float, N> &x)
{
  return cublasSasum(N, x, x.inc());
}

template <int N>
void axpy(float alpha, const Vector<float, N> &x, Vector<float, N> &y)
{
  cublasSaxpy(N, alpha, x, x.inc(), y, y.inc());
}

template <int N>
void copy(const Vector<float, N> &x, Vector<float, N> &y)
{
  cublasScopy(N, x, x.inc(), y, y.inc());
}

template <int N>
float dot(const Vector<float, N> &x, Vector<float, N> &y)
{
  return cublasSdot(N, x, x.inc(), y, y.inc());
}

template <int N>
float nrm2(const Vector<float, N> &x)
{
  return cublasSnrm2(N, x, x.inc());
}

template <int N>
void scal(float alpha, const Vector<float, N> &x)
{
  cublasSscal(N, alpha, x, x.inc());
}

template <int N>
void swap(Vector<float, N> &x, Vector<float, N> &y)
{
  cublasSswap(N, x, x.inc(), y, y.inc());
}
