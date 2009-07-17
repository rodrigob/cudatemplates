#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include <cutil.h>
#include <cutil_math.h>
#include <cuda_runtime.h>


#define POLY_LENGTH 27


struct matrix4x4
{
  union
  {
    float m[16];
    float rc[4][4];
    struct
    {
      float4 r1;
      float4 r2;
      float4 r3;
      float4 r4;
    };
  };
};

inline __host__ __device__ matrix4x4 make_matrix4x4(float const * m)
{
  matrix4x4 mat;
  #pragma unroll
  for(int i = 0; i < 16; ++i)
  {
    mat.m[i] = m[i];
  }
  return mat;
}

inline __host__ __device__ float4 operator * (matrix4x4 const & m, float4 const & f)
{
  return make_float4(dot(m.r1,f), dot(m.r2,f), dot(m.r3,f),dot(m.r4,f));
}

inline __host__ __device__ matrix4x4 transpose(matrix4x4 const & m)
{
  matrix4x4 mat;
  #pragma unroll
  for(int i = 0; i < 4; ++i)
    #pragma unroll
    for(int j = 0; j < 4; ++j)
      mat.rc[i][j] = m.rc[j][i];
  return mat;
  
}


__constant__ float mvInverseMatrix[16] = { 1.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 1.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 1.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 1.0f } ;

__constant__ float bboxDimensions[3] = {1.0f, 1.0f, 1.0f};

__device__ bool check_boundingchange(float l, float3 const checknormal)
{
  if(dot(checknormal,checknormal) > 0)
    if(l>0)
      return false;

  return true;
}

__device__ int
d_isosurf_calc_BBLimits_and_normal(float2 position, float3* normal)
{
  float3 checknormal;
  float mvi41 = mvInverseMatrix[3*4+0];
  float mvi42 = mvInverseMatrix[3*4+1];
  float mvi43 = mvInverseMatrix[3*4+2];
  float mvi44 = mvInverseMatrix[3*4+3];

  matrix4x4 mv = transpose(make_matrix4x4(mvInverseMatrix));
  float w_amount = mvi44 +mvi41*position.x +mvi42*position.y;
  float z = bboxDimensions[2];
  float4 znormal = make_float4(0.0f,0.0f,1.0f,-z);
  float l = (z*w_amount-position.x-position.y);
  znormal = mv*znormal;
  checknormal = make_float3(znormal.x,znormal.y,znormal.z);

  if(check_boundingchange(l,checknormal) == false)
    return false;

  if(check_boundingchange(l,checknormal) == false)
    return false;

  return true;
}

__global__ void
d_isosurface_drawBB_global(float *depth, int *od, int w, int h)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int n = y * w + x;

  float2 position = make_float2(2.0f*(float)x/float(w)-1.0f,2.0f*(float)y/float(h)-1.0f);
  float3 normal;

  if(!d_isosurf_calc_BBLimits_and_normal(position, &normal))
    return;

  float z = 0.5f;

  if(depth[n] < 0)
    return;

  depth[n] = z;
}


template<size_t MAX_REC_DEPTH, typename node_info, size_t entrySize>
__device__ bool
process_subtree(float3 &t1_)
{
  int recursionDepth = 0;
  float3 t1_stack[MAX_REC_DEPTH];
  t1_stack[0] = t1_;

  while(1) {
    const float3 t1 = t1_stack[recursionDepth];

    if(t1.x < 0.0) {
      recursionDepth--;
      continue;
    }

    float3 exitpoint;
    t1_stack[recursionDepth] = exitpoint;
  }
}

template<size_t MAX_REC_DEPTH, typename node_info, size_t entrySize>
__device__ bool
process_octree()
{
  float3 t1;
  return process_subtree<MAX_REC_DEPTH, node_info, entrySize>(t1);
}

template<size_t coeff_count>
struct renderTreeNode
{
};

__device__ void
d_wavelet()
{
  process_octree<8,renderTreeNode<POLY_LENGTH>,sizeof(renderTreeNode<POLY_LENGTH>)>();
}

__global__ void
d_wavelet_global()
{
  d_wavelet();
}
