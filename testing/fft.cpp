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

#include <cstdlib>
#include <iostream>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/cufft.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/event.hpp>
#include <cudatemplates/hostmemoryheap.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif

using namespace std;


const size_t SIZE_    =  256;  // image size
const int    COUNT   = 1000;  // number of FFTs to perform
const float  EPSILON = 1e-5;  // error threshold


double
operator-(const struct timeval &t1, const struct timeval &t2)
{
  return (t1.tv_sec - t2.tv_sec) + (t1.tv_usec - t2.tv_usec) * 1e-6;
}

int
main()
{
  int err = 0;

  try {
    Cuda::Size<2> index;
    Cuda::HostMemoryHeap2D    <Cuda::FFT::real>    data1_h(SIZE_, SIZE_);
    Cuda::DeviceMemoryLinear2D<Cuda::FFT::real>    data1_g(SIZE_, SIZE_);
    Cuda::DeviceMemoryLinear2D<Cuda::FFT::complex> data_fft_g(SIZE_, SIZE_);  // can this be smaller?
    Cuda::DeviceMemoryLinear2D<Cuda::FFT::real>    data2_g(SIZE_, SIZE_);
    Cuda::HostMemoryHeap2D    <Cuda::FFT::real>    data2_h(SIZE_, SIZE_);

    // allocate memory:
    Cuda::FFT::Plan<Cuda::FFT::real, Cuda::FFT::complex, 2> plan_r2c_1d(data1_g.size);
    Cuda::FFT::Plan<Cuda::FFT::complex, Cuda::FFT::real, 2> plan_c2r_1d(data1_g.size);

    // create random data:
    for(index[0] = SIZE_; index[0]--;)
      for(index[1] = SIZE_; index[1]--;)
	data1_h[index] = rand() / (float)RAND_MAX;

    // copy data to device memory:
    copy(data1_g, data1_h);

    // execute FFT and measure performance:
    Cuda::Event t1, t2;
    t1.record();

    for(int i = COUNT; i--;) {
      plan_r2c_1d.exec(data1_g, data_fft_g);
      plan_c2r_1d.exec(data_fft_g, data2_g);
    }

    t2.record();
    t2.synchronize();
    double t = (t2 - t1) / 1000;
    cout
      << "total time: " << t << " seconds\n"
      << "FFTs per second (size = " << SIZE_ << "x" << SIZE_ << ", forward and inverse): " << (COUNT / t) << endl;

    // copy data to host memory:
    copy(data2_h, data2_g);

    // verify results:
    for(index[0] = SIZE_; index[0]--;) {
      for(index[1] = SIZE_; index[1]--;) {
	float d = data2_h[index] / (SIZE_ * SIZE_) - data1_h[index];

	if(fabs(d) > EPSILON) {
	  cerr << "FFT failed\n";
	  return 1;
	}
      }
    }
  }
  catch(const exception &e) {
    cerr << e.what();
    err = 1;
  }

  return err;
}
