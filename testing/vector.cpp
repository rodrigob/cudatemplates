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

#include <cudatemplates/size.hpp>
#include <cudatemplates/vector.hpp>

using namespace std;


/**
   Stream output operator.
   @param s output stream
   @param v vector
*/
template <class Type, unsigned Dim>
inline ostream &
operator<<(ostream &s, const Cuda::VectorBase<Type, Dim> &v)
{
  s << '(';

  for(unsigned i = 0; i < Dim; ++i) {
    if(i > 0)
      s << ',';

    s << v[i];
  }
  
  s << ')';
  return s;
}

int
main()
{
  Cuda::Vector<float, 1> v1a(1), v1b(2);
  Cuda::Vector<float, 2> v2a(3, 4), v2b(5, 6);
  Cuda::Vector<float, 3> v3a(7.5, 8.25, 9.125), v3b(10, 11, 12);
  Cuda::Vector<int, 3> v3c(13, 14, 15);

  cout << v1a << (v1a + v1b) << endl;
  cout << v2a << (v2a + v2b) << endl;
  cout << v3a << (v3a + v3b) << endl;

  // operator with type promotion and assignment with conversion:
  v3b = v3a + v3c;
  v3c = v3c + v3a;
  cout << v3b << endl;
  cout << v3c << endl;

  cout << v1a[0] << ' ' << v2a[1] << ' ' << v3a[2] << endl;

  Cuda::Size<1> s1a(1), s1b(2);
  Cuda::Size<2> s2a(3, 4), s2b(5, 6);
  Cuda::Size<3> s3a(7, 8, 9), s3b(10, 11, 12);

  cout << s1a[0] << ' ' << s2a[1] << ' ' << s3a[2] << endl;

  // Cuda::Size<3> s3c = s3a + s3b;
  s1a[0] = 12345;
  cout << s1a[0] << endl;
  cout << ((Cuda::SizeBase<1> &)(s1a))[0] << endl;
  cout << ((Cuda::VectorBase<size_t, 1> &)(s1a))[0] << endl;
  cout << ((Cuda::Vector<size_t, 1> &)(s1a))[0] << endl;

  cout << s3a.getSize() << endl;

  cout << sizeof(Cuda::VectorBase<float, 100>) << endl;
  cout << sizeof(Cuda::Vector<float, 100>) << endl;

  return 0;
}
