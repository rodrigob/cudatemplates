__global__ void transferInterleavedKernel( float4* input, float4* output,
                                           int width, int height, int p)
{
  // calculate absolute coordinates
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int c = y*p+x;

  // Thread index
  int tx = threadIdx.x+1;
  int ty = threadIdx.y+1;

  // Define arrays for shared memory
  __shared__ float data_shared[16+2][16+2][3];

  // load data into shared memory
  float4 temp = input[c];
  data_shared[ty][tx][0] = temp.x;
  data_shared[ty][tx][1] = temp.y;
  data_shared[ty][tx][2] = temp.z;

  __syncthreads();

  if (x == 0)
  {
    data_shared[ty][tx-1][0] = 0.0f;
    data_shared[ty][tx-1][1] = 0.0f;
    data_shared[ty][tx-1][2] = 0.0f;
  }
  else if (tx == 1)
  {
    temp = input[c-1];
    data_shared[ty][tx-1][0] = temp.x;
    data_shared[ty][tx-1][1] = temp.y;
    data_shared[ty][tx-1][2] = temp.z;
  }

  if (y == 0)
  {
    data_shared[ty-1][tx][0] = 0.0f;
    data_shared[ty-1][tx][1] = 0.0f;
    data_shared[ty-1][tx][2] = 0.0f;
  }
  else if (ty == 1)
  {
    temp = input[c-p];
    data_shared[ty-1][tx][0] = temp.x;
    data_shared[ty-1][tx][1] = temp.y;
    data_shared[ty-1][tx][2] = temp.z;
  }

  if (x >= width-1)
  {
    data_shared[ty][tx+1][0] = data_shared[ty][tx][0];
    data_shared[ty][tx+1][1] = data_shared[ty][tx][1];
    data_shared[ty][tx+1][2] = data_shared[ty][tx][2];
  }
  else if (tx == 16-1)
  {
    temp = input[c-p];
    data_shared[ty][tx+1][0] = temp.x;
    data_shared[ty][tx+1][1] = temp.y;
    data_shared[ty][tx+1][2] = temp.z;
  }

  if (y >= height-1)
  {
    data_shared[ty+1][tx][0] = data_shared[ty][tx][0];
    data_shared[ty+1][tx][1] = data_shared[ty][tx][1];
    data_shared[ty+1][tx][2] = data_shared[ty][tx][2];
  }
  else if (ty == 16-1)
  {
    temp = input[c+p];
    data_shared[ty+1][tx][0] = temp.x;
    data_shared[ty+1][tx][1] = temp.y;
    data_shared[ty+1][tx][2] = temp.z;
  }

  if ((x<width) && (y<height))
  {
    output[c] = make_float4(data_shared[ty][tx][0], data_shared[ty][tx][1], data_shared[ty][tx][2], 1.0f);
  }
}


__global__ void transferPlaneKernel( float* input, float* output,
                                     int width, int height, int p, int pitchY)
{
  // calculate absolute coordinates
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int c = y*p+x;

  // Thread index
  int tx = threadIdx.x+1;
  int ty = threadIdx.y+1;

  // Define arrays for shared memory
  __shared__ float data_shared[16+2][16+2][3];

  // load data into shared memory
  data_shared[ty][tx][0] = input[c];
  data_shared[ty][tx][1] = input[c+pitchY];
  data_shared[ty][tx][2] = input[c+2*pitchY];

  __syncthreads();

  if (x == 0)
  {
    data_shared[ty][tx-1][0] = 0.0f;
    data_shared[ty][tx-1][1] = 0.0f;
    data_shared[ty][tx-1][2] = 0.0f;
  }
  else if (tx == 1)
  {
    data_shared[ty][tx-1][0] = input[c-1];
    data_shared[ty][tx-1][1] = input[c-1+pitchY];
    data_shared[ty][tx-1][2] = input[c-1+2*pitchY];
  }

  if (y == 0)
  {
    data_shared[ty-1][tx][0] = 0.0f;
    data_shared[ty-1][tx][1] = 0.0f;
    data_shared[ty-1][tx][2] = 0.0f;
  }
  else if (ty == 1)
  {
    data_shared[ty-1][tx][0] = input[c-p];
    data_shared[ty-1][tx][1] = input[c-p+pitchY];
    data_shared[ty-1][tx][2] = input[c-p+2*pitchY];
  }

  if (x >= width-1)
  {
    data_shared[ty][tx+1][0] = data_shared[ty][tx][0];
    data_shared[ty][tx+1][1] = data_shared[ty][tx][1];
    data_shared[ty][tx+1][2] = data_shared[ty][tx][2];
  }
  else if (tx == 16-1)
  {
    data_shared[ty][tx+1][0] = input[c-p];
    data_shared[ty][tx+1][1] = input[c-p+pitchY];
    data_shared[ty][tx+1][2] = input[c-p+2*pitchY];
  }

  if (y >= height-1)
  {
    data_shared[ty+1][tx][0] = data_shared[ty][tx][0];
    data_shared[ty+1][tx][1] = data_shared[ty][tx][1];
    data_shared[ty+1][tx][2] = data_shared[ty][tx][2];
  }
  else if (ty == 16-1)
  {
    data_shared[ty+1][tx][0] = input[c+p];
    data_shared[ty+1][tx][1] = input[c+p+pitchY];
    data_shared[ty+1][tx][2] = input[c+p+2*pitchY];
  }

  if ((x<width) && (y<height))
  {
    output[c] = data_shared[ty][tx][0];
    output[c+pitchY] = data_shared[ty][tx][0];
    output[c+2*pitchY] = data_shared[ty][tx][0];
  }
}