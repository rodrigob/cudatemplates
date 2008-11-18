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
int iamax(const Vector<double, N> &x)
{
  return cublasIdamax(N, x, x.inc());
}

template <int N>
int iamin(const Vector<double, N> &x)
{
  return cublasIdamin(N, x, x.inc());
}

template <int N>
double asum(const Vector<double, N> &x)
{
  return cublasDasum(N, x, x.inc());
}

template <int N>
void axpy(double alpha, const Vector<double, N> &x, Vector<double, N> &y)
{
  cublasDaxpy(N, alpha, x, x.inc(), y, y.inc());
}

template <int N>
void copy(const Vector<double, N> &x, Vector<double, N> &y)
{
  cublasDcopy(N, x, x.inc(), y, y.inc());
}

template <int N>
double dot(const Vector<double, N> &x, Vector<double, N> &y)
{
  return cublasDdot(N, x, x.inc(), y, y.inc());
}

template <int N>
double nrm2(const Vector<double, N> &x)
{
  return cublasDnrm2(N, x, x.inc());
}

template <int N>
void scal(double alpha, const Vector<double, N> &x)
{
  cublasDscal(N, alpha, x, x.inc());
}

template <int N>
void swap(Vector<double, N> &x, Vector<double, N> &y)
{
  cublasDswap(N, x, x.inc(), y, y.inc());
}
