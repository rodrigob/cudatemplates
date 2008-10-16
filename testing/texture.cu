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


#include "texture_kernel.cu"


int
main()
{
#ifdef _DEVICEEMU

  fprintf(stderr,
	  "This file terminates with a \"segmentation fault\" in device emulation mode\n"
	  "(don't know why, if you find out, please let me know - Markus)\n");
  return 1;

#else

  // allocate memory:
  memhost_t hbuf1(SIZE, SIZE), hbuf2(SIZE, SIZE);
  memdev_t dbuf(SIZE, SIZE);
  array_t array(SIZE, SIZE);

  // create random image:
  Cuda::Size<2> index1, index2;

  for(index1[1] = SIZE; index1[1]--;)
    for(index1[0] = SIZE; index1[0]--;)
      hbuf1[index1] = (float)(rand() % RANGE);

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

  for(index1[1] = SIZE; index1[1]--;) {
    index2[1] = SIZE - 1 - index1[1];

    for(index1[0] = SIZE; index1[0]--;) {
      index2[0] = SIZE - 1 - index1[0];

      if(hbuf1[index1] != hbuf2[index2]) {
	fprintf(stderr, "texture access error\n");
	return 1;
      }
    }
  }

  return 0;

#endif
}
