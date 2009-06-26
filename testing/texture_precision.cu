#include <stdio.h>

#include <cudatemplates/array.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/hostmemoryreference.hpp>


Cuda::Array1D<float>::Texture tex;


float texdata[] = { 1, 2 };

const int NUM_SAMPLES = 1024;


// interpolate by means of texture unit:
__global__ void
kernel(int size, Cuda::DeviceMemoryLinear1D<float>::KernelData data)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  float t = (float)i / (gridDim.x * blockDim.x) * size;
  data.data[i] = tex1D(tex, t);
}

int
main()
{
  // allocate memory:
  Cuda::Size<1> size_texture(sizeof(texdata) / sizeof(texdata[0])), size_data(NUM_SAMPLES);
  Cuda::HostMemoryReference1D<float> h_texture(size_texture, texdata);
  Cuda::HostMemoryHeap1D<float> h_data(size_data);
  Cuda::Array1D<float> a_texture(size_texture);
  Cuda::DeviceMemoryLinear1D<float> d_data(size_data);
  
  // copy texture to GPU:
  Cuda::copy(a_texture, h_texture);
  tex.normalized = 0;
  tex.filterMode = cudaFilterModeLinear;
  tex.addressMode[0] = cudaAddressModeClamp;
  a_texture.bindTexture(tex);

  // execute kernel:
  dim3 blockDim(256);
  dim3 gridDim(size_data[0] / blockDim.x);
  kernel<<<gridDim, blockDim>>>(size_texture[0], d_data);

  // copy data to CPU:
  Cuda::copy(h_data, d_data);

  printf("index  GPU       CPU        diff\n");

  // report results:
  for(int i = 0; i < h_data.size[0]; ++i) {
    // read interpolated value computed on GPU:
    float t1 = h_data[i];

    // compute exact value on CPU:
    float t = (float)i / (gridDim.x * blockDim.x) * size_texture[0] - 0.5;
    int ti = floor(t);
    float tf = t - ti;
    float t2;

    if(ti < 0)
      t2 = texdata[0];
    else if(ti >= size_texture[0] - 1)
      t2 = texdata[size_texture[0] - 1];
    else
      t2 = (1 - tf) * texdata[ti] + tf * texdata[ti + 1];

    // print:
    printf("%4d: %9.6f %9.6f  %9.6f\n", i, t1, t2, t2 - t1);
  }

  return 0;
}
