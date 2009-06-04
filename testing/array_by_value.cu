#include <assert.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/copy_constant.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>


#define SIZE 32


/*
  This example demonstrates how an array can be "passed-by-value" to a Cuda
  kernel by embedding it into a struct.
*/

struct Param
{
  int data[SIZE];
};

__global__ void kernel(Param param, int *data_global)
{
  data_global[threadIdx.x] = param.data[threadIdx.x];
}

int
main()
{
  Cuda::HostMemoryHeap1D<int> h_data(SIZE);
  Cuda::DeviceMemoryLinear1D<int> d_data(SIZE);
  Param param;

  for(int i = SIZE; i--;)
    param.data[i] = rand();

  dim3 gridDim(1), blockDim(SIZE);
  kernel<<<gridDim, blockDim>>>(param, d_data.getBuffer());
  Cuda::copy(h_data, d_data);

  for(int i = SIZE; i--;)
    assert(h_data[i] == param.data[i]);
}
