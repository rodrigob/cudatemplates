#include <cstdlib>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>


typedef double real_t;
const size_t SIZE = 256;


__global__ void
kernel(Cuda::DeviceMemoryLinear1D<int>::KernelData buf)
{
  real_t e = 1;

  for(int i = threadIdx.x; i--;)
    e /= 2;

  real_t a = 1 + e;
  buf.data[threadIdx.x] = (a != 1) ? 1 : 0;
}

int
main()
{
  Cuda::HostMemoryHeap1D<int> hbuf(SIZE);
  Cuda::DeviceMemoryLinear1D<int> dbuf(SIZE);

  kernel<<<dim3(1, 1, 1), dim3(SIZE, 1, 1)>>>(dbuf);
  copy(hbuf, dbuf);

  unsigned i;

  for(i = 0; i < SIZE; ++i)
    if(hbuf[i] == 0)
      break;

  printf("%d bits precision\n", i);
}
