#include <iostream>

#include <cuda_runtime.h>

using namespace std;


/**
   This example does the following (all numeric values
   refer to all three dimensions):
   *) allocate array1 holding 32 elements
   *) allocate array2 holding 16 elements
   *) copy elements [12, 13, ..., 19] of array1
      to   elements [ 4,  5, ..., 11] of array2
   
   The copy operation fails with an "invalid argument" error,
   although all requested indices are within the allocated ranges.
   If size2 in increased to 20 such that the *source* indices
   fit into the *destination* array, the copy operation succeeds.

   When the destination array is smaller than the source array,
   cudaMemcpy3D() seems to check the *source* indices
   against the size of the *destination* array.
*/


int
main()
{
  const int size1 = 32;  // size of source array
  const int size2 = 16;  // size of destination array
  const int size3 =  8;  // size of region to be copied
  const int pos1  = 12;  // source offset
  const int pos2  =  4;  // destination offset

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaExtent extent1, extent2;
  cudaArray *array1, *array2;

  // create first CUDA array (32x32x32 elements):
  extent1.width  = size1;
  extent1.height = size1;
  extent1.depth  = size1;
  cudaMalloc3DArray(&array1, &channelDesc, extent1);

  // create second CUDA array (16x16x16 elements):
  extent2.width  = size2;
  extent2.height = size2;
  extent2.depth  = size2;
  cudaMalloc3DArray(&array2, &channelDesc, extent2);

  // copy elements [12, 13, ..., 19] of array1
  // to   elements [ 4,  5, ..., 11] of array2:
  cudaMemcpy3DParms params = { 0 };
  params.srcArray = array1;
  params.srcPos.x = pos1;
  params.srcPos.y = pos1;
  params.srcPos.z = pos1;
  params.dstArray = array2;
  params.dstPos.x = pos2;
  params.dstPos.y = pos2;
  params.dstPos.z = pos2;
  params.extent.width  = size3;
  params.extent.height = size3;
  params.extent.depth  = size3;
  params.kind = cudaMemcpyDeviceToDevice;
  cudaError_t ret = cudaMemcpy3D(&params);

  // print result:
  cout << cudaGetErrorString(ret) << endl;
  return ret;
}
