#include <cutil_math.h>

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

inline __device__ float4 operator * (matrix4x4 const & m, float4 const & f)
{
  return make_float4(dot(m.r1, f), dot(m.r2, f), dot(m.r3, f), dot(m.r4, f));
}

inline __device__ matrix4x4 operator *(matrix4x4 const & m1, matrix4x4 const & m2)
{
  matrix4x4 result;

  for(int i = 0; i < 4; ++i)
    for(int j = 0; j < 4; ++j) {
      result.rc[i][j] = 0.0f;
      for(int k = 0; k < 4; ++k)
        result.rc[i][j] += m1.rc[i][k] * m2.rc[k][j];
    }

  return result;
}

__global__ void
test(float4 *p)
{
  matrix4x4 m;
  float4 v;
  *p = m * m * v;
}

int
main()
{
}
