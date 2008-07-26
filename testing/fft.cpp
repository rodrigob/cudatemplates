#include <cstdlib>
#include <iostream>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/cufftpp.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>

using namespace std;

const float EPSILON = 1e-6;  // error threshold


int
main()
{
  const int SIZE = 1024;
  Cuda::HostMemoryHeap1D    <Cuda::FFT::real>    data1_h(SIZE);
  Cuda::DeviceMemoryLinear1D<Cuda::FFT::real>    data1_g(SIZE);
  Cuda::DeviceMemoryLinear1D<Cuda::FFT::complex> data_fft_g(SIZE / 2 + 1);
  Cuda::DeviceMemoryLinear1D<Cuda::FFT::real>    data2_g(SIZE);
  Cuda::HostMemoryHeap1D    <Cuda::FFT::real>    data2_h(SIZE);

  Cuda::FFT::Plan<Cuda::FFT::real, Cuda::FFT::complex, 1> plan_r2c_1d(data1_g.size);
  Cuda::FFT::Plan<Cuda::FFT::complex, Cuda::FFT::real, 1> plan_c2r_1d(data1_g.size);

  for(int i = SIZE; i--;)
    (data1_h.getBuffer())[i] = rand() / (float)RAND_MAX;

  copy(data1_g, data1_h);
  plan_r2c_1d.exec(data1_g, data_fft_g);
  plan_c2r_1d.exec(data_fft_g, data2_g);
  copy(data2_h, data2_g);

  for(int i = SIZE; i--;) {
    float d = (data2_h.getBuffer())[i] / SIZE - (data1_h.getBuffer())[i];

    if(fabs(d) > EPSILON) {
      cerr << "FFT failed\n";
      return 1;
    }
  }

  return 0;
}
