#include <cudatemplates/array.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>


const int SIZE = 256;
const int BLOCK_SIZE = 16;


typedef Cuda::HostMemoryHeap2D<float> memhost_t;
typedef Cuda::DeviceMemoryLinear2D<float> memdev_t;
typedef Cuda::Array2D<float> array_t;
// typedef array_t::Texture<cudaReadModeElementType> texture_t;
// typedef array_t::Texture texture_t;
// typedef texture<float, 2, cudaReadModeElementType> texture_t;


// texture_t tex;
// texture<float, 2, cudaReadModeElementType> tex;

/*
// __global__ void kernel(texture<float, 2, cudaReadModeElementType> tex, memdev_t::KernelData res)
__global__ void kernel(memdev_t::KernelData res)
{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  res.data[x + y * SIZE] = tex2D(tex, x, y);
}
*/

int
main()
{
  memhost_t h(SIZE, SIZE);
  memdev_t d(SIZE, SIZE);
  array_t a(SIZE, SIZE);
  copy(a, h);
  /*
  a.bindTexture(tex);

  // execute kernel:
  dim3 dimGrid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE, 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  kernel<<<dimGrid, dimBlock>>>(d);
  */
  return 0;
}
