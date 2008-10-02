#include <cudatemplates/array.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>


const int SIZE = 256;
const int BLOCK_SIZE = 16;
const int RANGE = 1000000;


typedef Cuda::HostMemoryHeap2D<float> memhost_t;
typedef Cuda::DeviceMemoryLinear2D<float> memdev_t;
typedef Cuda::Array2D<float> array_t;
typedef array_t::Texture texture_t;

texture_t tex;

__global__ void kernel(memdev_t::KernelData res)
{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  res.data[x + y * SIZE] = tex2D(tex, x + 0.5, y + 0.5);
}

int
main()
{
  // allocate memory:
  memhost_t hbuf1(SIZE, SIZE), hbuf2(SIZE, SIZE);
  memdev_t dbuf(SIZE, SIZE);
  array_t array(SIZE, SIZE);

  // create random image:
  Cuda::Size<2> index;

  for(index[1] = SIZE; index[1]--;)
    for(index[0] = SIZE; index[0]--;)
      hbuf1[index] = random() % RANGE;

  // copy image to array and bind array as texture:
  copy(array, hbuf1);
  tex.filterMode = cudaFilterModeLinear;
  array.bindTexture(tex);

  // execute kernel:
  dim3 dimGrid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE, 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  kernel<<<dimGrid, dimBlock>>>(dbuf);

  // verify:
  copy(hbuf2, dbuf);

  for(index[1] = SIZE; index[1]--;)
    for(index[0] = SIZE; index[0]--;)
      if(hbuf1[index] != hbuf2[index]) {
	fprintf(stderr, "texture access error\n");
	return 1;
      }

  return 0;
}
