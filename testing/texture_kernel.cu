/*
  The kernel is in a separate file to test if dependencies are handled correctly.
  It is meant to be included in "texture.cu" and not compiled as a standalone program.
*/

__global__ void kernel(memdev_t::KernelData res)
{
  CUDA_STATIC_ASSERT(1 > 0);
  CUDA_STATIC_ASSERT(2 > 0);

  int x1 = threadIdx.x + blockDim.x * blockIdx.x;
  int y1 = threadIdx.y + blockDim.y * blockIdx.y;
  int x2 = res.size[0] - 1 - x1;
  int y2 = res.size[1] - 1 - y1;
  res.data[x1 + y1 * SIZE] = tex2D(tex, x2 + 0.5, y2 + 0.5);
}
