//#include "../../include/cudatemplates/cuda_gcc43_compat.hpp"
#include <iostream>
#include <cutil.h>

__global__ void kernel1(float2* p, int dim)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
	
  int c = y*dim+x;

  p[c] = make_float2(1.0f,1.0f);
}

__global__ void kernel2(float2* p, float2 value, int dim)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
	
  int c = y*dim+x;

  p[c] = value;
}

__global__ void kernel3(float2* p, float2 value, int dim, float* dbg)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
	
  int c = y*dim+x;

  p[c] = value;

  *dbg = 1.0f;
}


int main()
{
  unsigned int dim = 16;
  unsigned int mem_size = sizeof(float2)*dim*dim;

  // allocate host memory
  float2* h_data = (float2*) malloc( mem_size);
  // initalize the memory
  for( unsigned int i = 0; i < dim*dim; ++i) 
    h_data[i] = make_float2(i,i);

  // Show content of device memory
  for (int y=0; y<dim; y++)
  {
    for (int x=0; x<dim; x++)
      std::cout << h_data[y*dim+x].x << "," << h_data[y*dim+x].y << " ";
    std::cout << std::endl;
  }   
  std::cout << std::endl;	

  // allocate device memory
  float2* d_data;
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_data, mem_size));
  // copy host memory to device
  CUDA_SAFE_CALL( cudaMemcpy( d_data, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // Prepare grid
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(1,1);
	
  ///////////////////////////////////////////////////////////////////////////////
  // Call first kernel
  kernel1<<<dimGrid, dimBlock>>>(d_data, dim);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Execution of kernel1 failed");
	
  // copy device memory back to host
  CUDA_SAFE_CALL( cudaMemcpy( h_data, d_data, mem_size,
                              cudaMemcpyDeviceToHost) );

  // Show content of device memory
  for (int y=0; y<dim; y++)
  {
    for (int x=0; x<dim; x++)
      std::cout << h_data[y*dim+x].x << "," << h_data[y*dim+x].y << " ";
    std::cout << std::endl;
  }   
  std::cout << std::endl;

  ///////////////////////////////////////////////////////////////////////////////
  // Call second kernel
  kernel2<<<dimGrid, dimBlock>>>(d_data, make_float2(0.0f, 0.0f), dim);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Execution of kernel2 failed");

  // copy device memory back to host
  CUDA_SAFE_CALL( cudaMemcpy( h_data, d_data, mem_size,
                              cudaMemcpyDeviceToHost) );

  // Show content of device memory
  for (int y=0; y<dim; y++)
  {
    for (int x=0; x<dim; x++)
      std::cout << h_data[y*dim+x].x << "," << h_data[y*dim+x].y << " ";
    std::cout << std::endl;
  }   
  std::cout << std::endl;

  ///////////////////////////////////////////////////////////////////////////////
  // allocate host memory
  float* h_dbg = (float*) malloc(sizeof(float));
  // allocate device memory for debug flag
  float* d_dbg;
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_dbg, sizeof(float)));
  // copy host memory to device
  CUDA_SAFE_CALL( cudaMemcpy( d_dbg, h_dbg, sizeof(float),
                              cudaMemcpyHostToDevice) );

  // Call third kernel
  kernel3<<<dimGrid, dimBlock>>>(d_data, make_float2(0.0f, 0.0f), dim, d_dbg);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Execution of kernel3 failed");

}
