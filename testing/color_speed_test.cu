#include <cudatemplates/cuda_gcc43_compat.hpp>

#include <iostream>

#include <cudatemplates/devicememorypitched.hpp>

#include <cuda_runtime.h>

#include <sys/time.h>

using namespace std;

#include "color_speed_test_kernels.cu"

/**
Does a speed test for color images
*/

double getTime()
{
  cudaThreadSynchronize();
  timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec * 1000.0 + time.tv_usec / 1000.0;
}

  inline unsigned int divUp(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
  }

int CUDAtestMemLoad(int block_size, int num, int width, int height)
{
  cout << "Testing Color Image loads" << endl;
  cout << "  Block size      = " << block_size << endl;
  cout << "  Number of calls = " << num << endl;
  cout << "  Size            = " << width << "x" << height << endl;

  Cuda::Size<2> interleaved_size(divUp(width, block_size)*block_size, divUp(height, block_size)*block_size);
  Cuda::Size<3> plane_size(divUp(width, block_size)*block_size, divUp(height, block_size)*block_size,3);
  Cuda::DeviceMemoryPitched<float4, 2> interleaved_image_in(interleaved_size);
  Cuda::DeviceMemoryPitched<float4, 2> interleaved_image_out(interleaved_size);
  Cuda::DeviceMemoryPitched<float, 3> plane_image_in(plane_size);
  Cuda::DeviceMemoryPitched<float, 3> plane_image_out(plane_size);
//   COMMONLIB_CHECK_CUDA_ERROR();

  interleaved_image_in.initMem(1);
  interleaved_image_out.initMem(1);
  plane_image_in.initMem(1);
  plane_image_out.initMem(1);
//   COMMONLIB_CHECK_CUDA_ERROR();

  // prepare fragmentation for processing
  dim3 dimBlock(block_size, block_size, 1);
  dim3 dimGrid(divUp(width, block_size), divUp(height, block_size), 1);

  cout << "float4 interleaved image          -  ";
  cudaThreadSynchronize();
  double start_time = getTime();
  for (int i=0; i<num; i++)
  {
    transferInterleavedKernel<<<dimGrid, dimBlock>>>( interleaved_image_in.getBuffer(),
                                                      interleaved_image_out.getBuffer(),
                                                      interleaved_image_in.region_size[0],
                                                      interleaved_image_in.region_size[1],
                                                      interleaved_image_in.stride[0]);
    CUDA_CHECK(cudaGetLastError());
    cudaThreadSynchronize();
  }
  cudaThreadSynchronize();
  cout << getTime() - start_time << endl;
//   COMMONLIB_CHECK_CUDA_ERROR();

  cout << "float4 interleaved image (direct) -  ";
  cudaThreadSynchronize();
  start_time = getTime();
  for (int i=0; i<num; i++)
  {
    transferInterleavedDirectKernel<<<dimGrid, dimBlock>>>( interleaved_image_in.getBuffer(),
                                                            interleaved_image_out.getBuffer(),
                                                            interleaved_image_in.region_size[0],
                                                            interleaved_image_in.region_size[1],
                                                            interleaved_image_in.stride[0]);
    CUDA_CHECK(cudaGetLastError());
    cudaThreadSynchronize();
  }
  cudaThreadSynchronize();
  cout << getTime() - start_time << endl;

  cout << "float 3-plane image               -  ";
  cudaThreadSynchronize();
  start_time = getTime();
  for (int i=0; i<num; i++)
  {
    transferPlaneKernel<<<dimGrid, dimBlock>>>( plane_image_in.getBuffer(),
                                                plane_image_in.getBuffer(),
                                                plane_image_in.region_size[0],
                                                plane_image_in.region_size[1],
                                                plane_image_in.stride[0],
                                                plane_image_in.stride[1]);

    CUDA_CHECK(cudaGetLastError());
    cudaThreadSynchronize();
  }
  cudaThreadSynchronize();
  cout << getTime() - start_time << endl;
//   COMMONLIB_CHECK_CUDA_ERROR();

  cout << endl << endl;
  return 0;
}

int
main()
{
  CUDAtestMemLoad(16, 5000, 512, 512);
  CUDAtestMemLoad(16, 5000, 327, 571);
  CUDAtestMemLoad(16, 5000, 34, 23);
  CUDAtestMemLoad(16, 5000, 949, 1003);

  return 0;
}

