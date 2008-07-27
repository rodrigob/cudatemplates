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

#define CUDA_NO_DEFAULT_CONSTRUCTORS
#define ENFORCE_LAYOUT 0
#define VERBOSE 0
#define TEST_REFERENCE 0

#include <cudatemplates/array.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/devicememoryreference.hpp>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/hostmemorylocked.hpp>
#include <cudatemplates/hostmemoryreference.hpp>

using namespace std;


static float
my_random()
{
  // return rand() % 100;
  return rand() / (float)RAND_MAX;
}

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
      const Cuda::Size<T1::Dim> &size, bool use_region)
{
  BOOST_STATIC_ASSERT(T1::Dim == T2::Dim);
  
  // extract some types and constants:
  typedef typename T1::Type Type;
  const unsigned Dim = T1::Dim;

  {
    // create objects to be tested:
    T1 obj1(size);
    Cuda::Layout<Type, Dim> layout(obj1);
    T2 obj2(layout);

    // allocate plain C arrays and create HostMemoryReferences to them:
    Type *buf1 = new Type[layout.getSize()];
    Type *buf2 = new Type[layout.getSize()];
    Type *buf3 = new Type[layout.getSize()];

#if ENFORCE_LAYOUT
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
      buf1[i] = my_random();

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

    delete[] buf1;
    delete[] buf2;
    delete[] buf3;
  }

  if(use_region) {
    T1 obj1(size1);
    T2 obj2(size2);

    // allocate plain C arrays and create HostMemoryReferences to them:
    Type *buf1 = new Type[obj1.getSize()];
    Type *buf2 = new Type[obj2.getSize()];
    Type *buf3 = new Type[obj2.getSize()];
    Type *buf4 = new Type[size.getSize()];
    Type *buf5 = new Type[size.getSize()];

    Cuda::HostMemoryReference<Type, Dim> ref1(size1, buf1);
    Cuda::HostMemoryReference<Type, Dim> ref2(size2, buf2);
    Cuda::HostMemoryReference<Type, Dim> ref3(size2, buf3);
    Cuda::HostMemoryReference<Type, Dim> ref4(size, buf4);
    Cuda::HostMemoryReference<Type, Dim> ref5(size, buf5);

    // create random data:
    for(unsigned i = 0; i < ref1.getSize(); ++i)
      buf1[i] = my_random();

    for(unsigned i = 0; i < ref2.getSize(); ++i)
      buf2[i] = my_random();

    // copy data and overwrite subwindow:
    copy(obj1, ref1);
    copy(obj2, ref2);
    copy(obj2, obj1, pos2, pos1, size);
    copy(ref3, obj2);
    T2 obj4(obj1, pos1, size);
    copy(ref4, obj4);
#if TEST_REFERENCE
    typename T1::Reference obj5(obj1, pos1, size);
    copy(ref5, obj5);
#endif

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

      Type x3 = buf3[ref3.getOffset(index)];

      if(x3 != x1) {
	cerr << "copy test failed at index " << index << " in \"" << __PRETTY_FUNCTION__ << "\"\n";
	return 1;
      }

      if(inside) {
	Type x4 = buf4[ref4.getOffset(index - pos2)];

	if(x4 != x1) {
	  cerr << "constructor test failed at index " << index << " in \"" << __PRETTY_FUNCTION__ << "\"\n";
	  return 1;
	}

#if TEST_REFERENCE
	Type x5 = buf5[ref5.getOffset(index - pos2)];

	if(x5 != x1) {
	  cerr << "reference test failed at index " << index << " in \"" << __PRETTY_FUNCTION__ << "\"\n";
	  return 1;
	}
#endif
      }
    }
    while(increment(index, size2));

    delete[] buf1;
    delete[] buf2;
    delete[] buf3;
    delete[] buf4;
    delete[] buf5;
  }

#if VERBOSE
  cout << "test succeeded in \"" << __PRETTY_FUNCTION__ << "\"\n";
#endif

  return 0;
}

template <class T1, class T2>
int
test2(const Cuda::Size<T1::Dim> &size1, const Cuda::Size<T1::Dim> &size2,
      const Cuda::Size<T1::Dim> &pos1, const Cuda::Size<T1::Dim> &pos2,
      const Cuda::Size<T1::Dim> &size, size_t size_max)
{
  BOOST_STATIC_ASSERT(T1::Dim == T2::Dim);

  int err = 0;

  // test with fixed size:
  err |= test1<T1, T2>(size1, size2, pos1, pos2, size, false);
  
  if(size_max > 0) {
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

#if VERBOSE
    cout << rsize1 << rsize2 << rpos1 << rpos2 << rsize << endl;
#endif

    err |= test1<T1, T2>(rsize1, rsize2, rpos1, rpos2, rsize, true);
  }

  return err;
}


int
main()
{
  //srand(time(0));
  int seed = rand();
  srand(seed);
  int err = 0;

  try {
    // one-dimensional data:
    size_t smax1 = 512;
    Cuda::Size<1>
      size1a(smax1), size1b(smax1),
      pos1a(smax1 / 16), pos1b(smax1 / 16),
      size1(smax1 / 2);

#include "test1d.cpp"

    // two-dimensional data:
    size_t smax2 = 512;
    Cuda::Size<2>
      size2a(smax2, smax2), size2b(smax2, smax2),
      pos2a(smax2 / 16, smax2 / 16), pos2b(smax2 / 8, smax2 / 8),
      size2(smax2 / 2, smax2 / 2);

#include "test2d.cpp"

    // three-dimensional data:
    size_t smax3 = 64;
    Cuda::Size<3>
      size3a(smax3, smax3, smax3), size3b(smax3, smax3, smax3),
      pos3a(smax3 / 4, smax3 / 4, smax3 / 4), pos3b(smax3 / 8, smax3 / 8, smax3 / 8),
      size3(smax3 / 2, smax3 / 2, smax3 / 2);

#include "test3d.cpp"

    // simple usage example:
    {
      using namespace Cuda;
      Size<2> size(256, 256);
      HostMemoryHeap<float, 2> cpu(size);
      DeviceMemoryLinear<float, 2> gpu(size);
      copy(gpu, cpu);
    }
  }
  catch(const exception &e) {
    cerr << e.what();
    err = 1;
  }

  if(err)
    cerr << "random seed was " << seed << endl;

  return err;
}
