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
#include <stdexcept>

#include <cuda.h>
#include <cutil.h>

#define CTK_NO_DEFAULT_CONSTRUCTORS

#include <cudatemplates/array.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemorylocked.hpp>
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/hostmemoryreference.hpp>

using namespace std;


static const int RAND_RANGE = 100;


template <unsigned Dim>
ostream &operator<<(ostream &s, const Cuda::SizeBase<Dim> &sz)
{
  s << '(' << sz[0];

  for(unsigned i = 1; i < Dim; ++i)
    s << ", " << sz[i];

  return s << ')';
}

template <unsigned Dim>
bool increment(Cuda::SizeBase<Dim> &index, const Cuda::SizeBase<Dim> &size)
{
  for(unsigned j = 0; j < Dim; ++j) {
    if(++index[j] < size[j])
      return true;
    
    index[j] = 0;
  }
  
  return false;
}

template <class T1, class T2>
int
test1(const Cuda::Size<T1::Dim> &size1, const Cuda::Size<T1::Dim> &size2,
      const Cuda::Size<T1::Dim> &pos1, const Cuda::Size<T1::Dim> &pos2,
      const Cuda::Size<T1::Dim> &size)
{
  BOOST_STATIC_ASSERT(T1::Dim == T2::Dim);

#define USE_LAYOUT 1

  // extract some types and constants:
  typedef typename T1::Type Type;
  const unsigned Dim = T1::Dim;

  {
    // create objects to be tested:
    T1 obj1(size);
    Cuda::Layout<Type, Dim> layout(obj1);
    T2 obj2(layout);

    // allocate plain C arrays and create HostMemoryReferences to them:
    Type buf1[layout.getSize()], buf2[layout.getSize()], buf3[layout.getSize()];

#if USE_LAYOUT
    Cuda::HostMemoryReference<Type, Dim> ref1(layout, buf1);
    Cuda::HostMemoryReference<Type, Dim> ref2(layout, buf2);
    Cuda::HostMemoryReference<Type, Dim> ref3(layout, buf3);
#else
    Cuda::HostMemoryReference<Type, Dim> ref1(size, buf1);
    Cuda::HostMemoryReference<Type, Dim> ref2(size, buf2);
    Cuda::HostMemoryReference<Type, Dim> ref3(size, buf3);
#endif

    // create random data:
    for(int i = layout.getSize(); i--;)
      buf1[i] = rand() % RAND_RANGE;

    // copy data forth and back:
    copy(obj1, ref1);
    copy(obj2, obj1);
    copy(ref2, obj2);
    T2 obj3(obj1);
    copy(ref3, obj3);

    // compare results:
    Cuda::Size<Dim> index;

    for(int i = Dim; i--;)
      index[i] = 0;

    do {
      Type x1 = buf1[ref1.getOffset(index)];
      Type x2 = buf2[ref2.getOffset(index)];
      Type x3 = buf3[ref3.getOffset(index)];

      if(x2 != x1) {
	cerr << "copy test failed at index " << index << " in \"" << __PRETTY_FUNCTION__ << "\"\n";
	return 1;
      }

      if(x3 != x1) {
	cerr << "constructor test failed at index " << index << " in \"" << __PRETTY_FUNCTION__ << "\"\n";
	return 1;
      }
    }
    while(increment(index, size));
  }

  {
    T1 obj1(size1);
    T2 obj2(size2);

    // allocate plain C arrays and create HostMemoryReferences to them:
    Type buf1[obj1.getSize()], buf2[obj2.getSize()], buf3[obj2.getSize()];

    Cuda::HostMemoryReference<Type, Dim> ref1(size1, buf1);
    Cuda::HostMemoryReference<Type, Dim> ref2(size2, buf2);
    Cuda::HostMemoryReference<Type, Dim> ref3(size2, buf3);

    // create random data:
    for(unsigned i = 0; i < ref1.getSize(); ++i)
      buf1[i] = rand() % RAND_RANGE;

    for(unsigned i = 0; i < ref2.getSize(); ++i)
      buf2[i] = rand() % RAND_RANGE;

    // copy data and overwrite subwindow:
    copy(obj1, ref1);
    copy(obj2, ref2);
    copy(obj2, obj1, pos2, pos1, size);
    copy(ref3, obj2);

    // compare results:
    Cuda::Size<Dim> index;

    for(int i = Dim; i--;)
      index[i] = 0;

    do {
      // check if current index is inside copied area:
      bool inside = true;

      for(int i = Dim; i--;)
	if((index[i] < pos2[i]) || (index[i] >= pos2[i] + size[i])) {
	  inside = false;
	  break;
	}

      Type x1 = inside ?
	buf1[ref1.getOffset(index + pos1 - pos2)] :
	buf2[ref2.getOffset(index)];

      Type x2 = buf3[ref3.getOffset(index)];

      if(x2 != x1) {
	cerr << "copy test failed at index " << index << " in \"" << __PRETTY_FUNCTION__ << "\"\n";
	return 1;
      }
    }
    while(increment(index, size2));
  }

  // cout << "test succeeded in \"" << __PRETTY_FUNCTION__ << "\"\n";
  return 0;

#undef USE_LAYOUT

}

template <class T1, class T2>
int
test2(const Cuda::Size<T1::Dim> &size1, const Cuda::Size<T1::Dim> &size2,
      const Cuda::Size<T1::Dim> &pos1, const Cuda::Size<T1::Dim> &pos2,
      const Cuda::Size<T1::Dim> &size)
{
  BOOST_STATIC_ASSERT(T1::Dim == T2::Dim);

  // cout << "size1=" << size1 << ", size2=" << size2 << ", pos1=" << pos1 << ", pos2=" << pos2 << ", size=" << size << endl;

  int err = 0;
  err |= test1<T1, T1>(size1, size2, pos1, pos2, size);
  err |= test1<T1, T2>(size1, size2, pos1, pos2, size);
  err |= test1<T2, T1>(size1, size2, pos1, pos2, size);
  err |= test1<T2, T2>(size1, size2, pos1, pos2, size);
  return err;
}

template <class T1, class T2>
int
test3(const Cuda::Size<T1::Dim> &size1, const Cuda::Size<T1::Dim> &size2,
      const Cuda::Size<T1::Dim> &pos1, const Cuda::Size<T1::Dim> &pos2,
      const Cuda::Size<T1::Dim> &size, size_t size_max)
{
  BOOST_STATIC_ASSERT(T1::Dim == T2::Dim);

  int err = 0;

  // test with fixed size:
  err |= test2<T1, T2>(size1, size2, pos1, pos2, size);
  
  // test with random size:
  Cuda::Size<T1::Dim> rsize1, rsize2, rpos1, rpos2, rsize;

  for(int i = T1::Dim; i--;) {
    rsize1[i] = (rand() % (size_max - 1)) + 1;
    rsize2[i] = (rand() % (size_max - 1)) + 1;
    size_t m = min(rsize1[i], rsize2[i]);
    rsize[i] = (m > 1) ? (rand() % (m - 1)) + 1 : 1;
    size_t d1 = rsize1[i] - rsize[i];
    rpos1[i] = (d1 > 0) ? (rand() % d1) : 0;
    size_t d2 = rsize2[i] - rsize[i];
    rpos2[i] = (d2 > 0) ? (rand() % d2) : 0;
  }

  err |= test2<T1, T2>(rsize1, rsize2, rpos1, rpos2, rsize);
  return err;
}


int
main()
{
  srand(time(0));
  int seed = rand();
  srand(seed);

  try {
    int err = 0;

    // example for presentation:
    {
      using namespace Cuda;
      Size<2> size(256, 256);
      HostMemoryHeap<float, 2> cpu(size);
      DeviceMemoryLinear<float, 2> gpu(size);
      copy(gpu, cpu);
    }

    // one-dimensional data:
    size_t smax1 = 512;
    Cuda::Size<1>
      size1a(smax1), size1b(smax1),
      pos1a(smax1 / 16), pos1b(smax1 / 16),
      size1(smax1 / 2);
    err |= test3<Cuda::HostMemoryHeap<float, 1>, Cuda::DeviceMemoryLinear<float, 1> >(size1a, size1b, pos1a, pos1b, size1, smax1);
    err |= test3<Cuda::HostMemoryLocked<float, 1>, Cuda::DeviceMemoryLinear<float, 1> >(size1a, size1b, pos1a, pos1b, size1, smax1);

#if 0
    // not yet implemented:
    err |= test2<Cuda::HostMemoryHeap<float, 1>, Cuda::Array<float, 1> >(size1);
    err |= test2<Cuda::DeviceMemoryLinear<float, 1>, Cuda::Array<float, 1> >(size1);
#endif

    // two-dimensional data:
    size_t smax2 = 512;
    Cuda::Size<2>
      size2a(smax2, smax2), size2b(smax2, smax2),
      pos2a(smax2 / 16, smax2 / 16), pos2b(smax2 / 8, smax2 / 8),
      size2(smax2 / 2, smax2 / 2);
    err |= test3<Cuda::HostMemoryHeap<float, 2>, Cuda::DeviceMemoryLinear<float, 2> >(size2a, size2b, pos2a, pos2b, size2, smax2);
    err |= test3<Cuda::HostMemoryHeap<float, 2>, Cuda::DeviceMemoryPitched<float, 2> >(size2a, size2b, pos2a, pos2b, size2, smax2);
    err |= test3<Cuda::HostMemoryLocked<float, 2>, Cuda::DeviceMemoryLinear<float, 2> >(size2a, size2b, pos2a, pos2b, size2, smax2);
    err |= test3<Cuda::HostMemoryLocked<float, 2>, Cuda::DeviceMemoryPitched<float, 2> >(size2a, size2b, pos2a, pos2b, size2, smax2);
    err |= test3<Cuda::HostMemoryLocked<float, 2>, Cuda::Array<float, 2> >(size2a, size2b, pos2a, pos2b, size2, smax2);
    err |= test3<Cuda::HostMemoryHeap<float, 2>, Cuda::Array<float, 2> >(size2a, size2b, pos2a, pos2b, size2, smax2);
    err |= test3<Cuda::DeviceMemoryLinear<float, 2>, Cuda::Array<float, 2> >(size2a, size2b, pos2a, pos2b, size2, smax2);
    err |= test3<Cuda::DeviceMemoryPitched<float, 2>, Cuda::Array<float, 2> >(size2a, size2b, pos2a, pos2b, size2, smax2);

    // three-dimensional data:
    size_t smax3 = 64;
    Cuda::Size<3>
      size3a(smax3, smax3, smax3), size3b(smax3, smax3, smax3),
      pos3a(smax3 / 4, smax3 / 4, smax3 / 4), pos3b(smax3 / 8, smax3 / 8, smax3 / 8),
      size3(smax3 / 2, smax3 / 2, smax3 / 2);
    err |= test2<Cuda::HostMemoryHeap<float, 3>, Cuda::DeviceMemoryLinear<float, 3> >(size3a, size3b, pos3a, pos3b, size3/*, smax3*/);
    err |= test3<Cuda::HostMemoryHeap<float, 3>, Cuda::DeviceMemoryPitched<float, 3> >(size3a, size3b, pos3a, pos3b, size3, smax3);
    err |= test2<Cuda::HostMemoryLocked<float, 3>, Cuda::DeviceMemoryLinear<float, 3> >(size3a, size3b, pos3a, pos3b, size3/*, smax3*/);
    err |= test3<Cuda::HostMemoryLocked<float, 3>, Cuda::DeviceMemoryPitched<float, 3> >(size3a, size3b, pos3a, pos3b, size3, smax3);
    err |= test2<Cuda::HostMemoryLocked<float, 3>, Cuda::Array<float, 3> >(size3a, size3b, pos3a, pos3b, size3/*, smax3*/);
    err |= test2<Cuda::HostMemoryHeap<float, 3>, Cuda::Array<float, 3> >(size3a, size3b, pos3a, pos3b, size3/*, smax3*/);
    err |= test2<Cuda::DeviceMemoryLinear<float, 3>, Cuda::Array<float, 3> >(size3a, size3b, pos3a, pos3b, size3/*, smax3*/);
    err |= test2<Cuda::DeviceMemoryPitched<float, 3>, Cuda::Array<float, 3> >(size3a, size3b, pos3a, pos3b, size3/*, smax3*/);
    return err;
  }
  catch(const exception &e) {
    cerr << e.what() << endl;
    return 1;
  }
}
