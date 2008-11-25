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


void gemv(char type, complex alpha, const Matrix<complex> &A, const Vector<complex> &x, complex beta, Vector<complex> &y)
{
  assert(x.getSize() == y.getSize());
  assert(x.getSize() == A.getWidth());
  cublasCgemv(type, A.getHeight(), A.getWidth(), alpha, A, x.getSize(), x, x.inc(), beta, y, y.inc());
}
