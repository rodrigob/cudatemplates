#include <stdio.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>


typedef Cuda::HostMemoryHeap1D<float> memhost_t;
typedef Cuda::DeviceMemoryLinear1D<float> memdev_t;


__global__ void kernel(memdev_t::KernelData arg1,
		       memdev_t::KernelData arg2,
		       memdev_t::KernelData sum)
{
  sum.data[threadIdx.x] = arg1.data[threadIdx.x] + arg2.data[threadIdx.x];
}


int
main()
{
  // allocate CPU and GPU memory:
  const size_t SIZE = 256;
  memhost_t h_arg1(SIZE), h_arg2(SIZE), h_sum(SIZE);
  memdev_t d_arg1(SIZE), d_arg2(SIZE), d_sum(SIZE);

  // initialize data:
  const int RANGE = 1000000;

  for(int i = SIZE; i--;) {
    h_arg1[i] = random() % RANGE;
    h_arg2[i] = random() % RANGE;
  }

  // copy data from CPU to GPU memory:
  copy(d_arg1, h_arg1);
  copy(d_arg2, h_arg2);

  // execute kernel:
  dim3 dimGrid(1, 1, 1);
  dim3 dimBlock(SIZE, 1, 1);
  kernel<<<dimGrid, dimBlock>>>(d_arg1, d_arg2, d_sum);

  // copy data from GPU to CPU memory:
  copy(h_sum, d_sum);

  // verify results:
  for(int i = SIZE; i--;) {
    if(h_sum[i] != h_arg1[i] + h_arg2[i]) {
      fprintf(stderr, "failed\n");
      return 1;
    }
  }

  printf("ok\n");
  return 0;
}
