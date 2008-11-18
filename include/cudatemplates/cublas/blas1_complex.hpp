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
int iamax(const Vector<complex, N> &x)
{
  return cublasIcamax(N, x, x.inc());
}

template <int N>
int iamin(const Vector<complex, N> &x)
{
  return cublasIcamin(N, x, x.inc());
}

template <int N>
complex asum(const Vector<complex, N> &x)
{
  return cublasCasum(N, x, x.inc());
}

template <int N>
void axpy(complex alpha, const Vector<complex, N> &x, Vector<complex, N> &y)
{
  cublasCaxpy(N, alpha, x, x.inc(), y, y.inc());
}

template <int N>
void copy(const Vector<complex, N> &x, Vector<complex, N> &y)
{
  cublasCcopy(N, x, x.inc(), y, y.inc());
}

template <int N>
complex dot(const Vector<complex, N> &x, Vector<complex, N> &y)
{
  return cublasCdot(N, x, x.inc(), y, y.inc());
}

template <int N>
complex nrm2(const Vector<complex, N> &x)
{
  return cublasCnrm2(N, x, x.inc());
}

template <int N>
void scal(complex alpha, const Vector<complex, N> &x)
{
  cublasCscal(N, alpha, x, x.inc());
}

template <int N>
void swap(Vector<complex, N> &x, Vector<complex, N> &y)
{
  cublasCswap(N, x, x.inc(), y, y.inc());
}
