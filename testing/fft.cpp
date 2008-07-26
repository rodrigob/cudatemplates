#include <cudatemplates/copy.hpp>
#include <cudatemplates/cufftpp.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>


int
main()
{
  using namespace Cuda;
  using namespace FFT;

  const int SIZE = 1024;
  HostMemoryHeap1D    <real>    data1_h(SIZE);
  DeviceMemoryLinear1D<real>    data1_g(SIZE);
  DeviceMemoryLinear1D<complex> data_fft_g(SIZE / 2 + 1);
  DeviceMemoryLinear1D<real>    data2_g(SIZE);
  HostMemoryHeap1D    <real>    data2_h(SIZE);

  Plan<real, complex, 1> plan_r2c_1d(data1_g.size);
  Plan<complex, real, 1> plan_c2r_1d(data1_g.size);

  for(int i = SIZE; i--;)
    (data1_h.getBuffer())[i] = i;

  copy(data1_g, data1_h);
  plan_r2c_1d.exec(data1_g, data_fft_g);
  plan_c2r_1d.exec(data_fft_g, data2_g);
  copy(data2_h, data2_g);
}
