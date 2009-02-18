#include <iostream>

#include <cudatemplates/devicememorypitched.hpp>

#include <cuda_runtime.h>

using namespace std;


/**
Does a speed test for color images
*/

int CUDAtestMemLoad(int block_size, int num, int width, int height)
{
  cout << "Testing Color Image loads" << endl;
  cout << "  Block size      = " << block_size << endl;
  cout << "  Number of calls = " << num << endl;
  cout << "  Size            = " << width << "x" << height << endl;

  Cuda::Size<2> interleaved_size(width,height);
  Cuda::Size<3> plane_size(width,height,3);
  Cuda::DeviceMemoryPitched<float4, 2> interleaved_image_in(interleaved_size);
  Cuda::DeviceMemoryPitched<float4, 2> interleaved_image_out(interleaved_size);
  Cuda::DeviceMemoryPitched<float, 3> plane_image_in(plane_size);
  Cuda::DeviceMemoryPitched<float, 3> plane_image_out(plane_size);
  COMMONLIB_CHECK_CUDA_ERROR();

  interleaved_image_in.initMem(1);
  interleaved_image_out.initMem(1);
  plane_image_in.initMem(1);
  plane_image_out.initMem(1);
  COMMONLIB_CHECK_CUDA_ERROR();

  // prepare fragmentation for processing
  dim3 dimBlock(block_size, block_size, 1);
  dim3 dimGrid(CommonLib::divUp(interleaved_size[0], block_size), CommonLib::divUp(interleaved_size[1], block_size), 1);

  cout << "First Test: float4 interleaved image  -  ";
  cudaThreadSynchronize();
  double start_time = CommonLib::getTime();
  for (int i=0; i<num; i++)
  {
    transferInterleavedKernel<<<dimGrid, dimBlock>>>( interleaved_image_in.getBuffer(),
                                                      interleaved_image_out.getBuffer(),
                                                      interleaved_image_in.region_size[0],
                                                      interleaved_image_in.region_size[1],
                                                      interleaved_image_in.stride[0]);
                                                      cudaThreadSynchronize();
  }
  cudaThreadSynchronize();
  cout << CommonLib::getTime() - start_time << endl;
  COMMONLIB_CHECK_CUDA_ERROR();

  cout << "Second Test: float 3 plane image  -  ";
  cudaThreadSynchronize();
  start_time = CommonLib::getTime();
  for (int i=0; i<num; i++)
  {
    transferPlaneKernel<<<dimGrid, dimBlock>>>( plane_image_in.getBuffer(),
                                                plane_image_in.getBuffer(),
                                                plane_image_in.region_size[0],
                                                plane_image_in.region_size[1],
                                                plane_image_in.stride[0],
                                                plane_image_in.stride[1]);
                                                cudaThreadSynchronize();
  }
  cudaThreadSynchronize();
  cout << CommonLib::getTime() - start_time << endl;
  COMMONLIB_CHECK_CUDA_ERROR();


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
