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


int iamax(const Vector<complex> &x)
{
  return cublasIcamax(x.getSize(), x, x.inc());
}

int iamin(const Vector<complex> &x)
{
  return cublasIcamin(x.getSize(), x, x.inc());
}

float asum(const Vector<complex> &x)
{
  return cublasScasum(x.getSize(), x, x.inc());
}

void axpy(complex alpha, const Vector<complex> &x, Vector<complex> &y)
{
  assert(x.getSize() == y.getSize()); 
  cublasCaxpy(x.getSize(), alpha, x, x.inc(), y, y.inc());
}

void copy(const Vector<complex> &x, Vector<complex> &y)
{
  assert(x.getSize() == y.getSize()); 
  cublasCcopy(x.getSize(), x, x.inc(), y, y.inc());
}

complex dotc(const Vector<complex> &x, const Vector<complex> &y)
{
  assert(x.getSize() == y.getSize()); 
  return cublasCdotc(x.getSize(), x, x.inc(), y, y.inc());
}

float nrm2(const Vector<complex> &x)
{
  return cublasScnrm2(x.getSize(), x, x.inc());
}

void scal(complex alpha, Vector<complex> &x)
{
  cublasCscal(x.getSize(), alpha, x, x.inc());
}

void swap(Vector<complex> &x, Vector<complex> &y)
{
  assert(x.getSize() == y.getSize()); 
  cublasCswap(x.getSize(), x, x.inc(), y, y.inc());
}
