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


int iamax(const Vector<double> &x)
{
  return cublasIdamax(x.getSize(), x, x.inc());
}

int iamin(const Vector<double> &x)
{
  return cublasIdamin(x.getSize(), x, x.inc());
}

double asum(const Vector<double> &x)
{
  return cublasDasum(x.getSize(), x, x.inc());
}

void axpy(double alpha, const Vector<double> &x, Vector<double> &y)
{
  assert(x.getSize() == y.getSize()); 
  cublasDaxpy(x.getSize(), alpha, x, x.inc(), y, y.inc());
}

void copy(const Vector<double> &x, Vector<double> &y)
{
  assert(x.getSize() == y.getSize()); 
  cublasDcopy(x.getSize(), x, x.inc(), y, y.inc());
}

double dot(const Vector<double> &x, const Vector<double> &y)
{
  assert(x.getSize() == y.getSize()); 
  return cublasDdot(x.getSize(), x, x.inc(), y, y.inc());
}

double nrm2(const Vector<double> &x)
{
  return cublasDnrm2(x.getSize(), x, x.inc());
}

void scal(double alpha, Vector<double> &x)
{
  cublasDscal(x.getSize(), alpha, x, x.inc());
}

void swap(Vector<double> &x, Vector<double> &y)
{
  assert(x.getSize() == y.getSize()); 
  cublasDswap(x.getSize(), x, x.inc(), y, y.inc());
}
