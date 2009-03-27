/*

Compiling this file with "nvcc bug3.cu" gives the following error message:

--------------------------------------------------------------------------------
Signal: Segmentation fault in Global Optimization -- Expression Reshaping phase.
<input>(0): Error: Signal Segmentation fault in phase Global Optimization -- Expression Reshaping -- processing aborted
*** Internal stack backtrace:
    /usr/open64/lib//be [0x6a342f]
    /usr/open64/lib//be [0x6a4079]
    /usr/open64/lib//be [0x6a37cd]
    /usr/open64/lib//be [0x6a4a16]
    /lib64/libc.so.6 [0x2ac9462a76e0]
    /usr/open64/lib//be [0x40f3f8]
    /usr/open64/lib//be [0x411c25]
    /usr/open64/lib//be [0x411f45]
    /usr/open64/lib//be [0x411ffd]
    /usr/open64/lib//be [0x4138ce]
    /usr/open64/lib//be [0x4df593]
    /usr/open64/lib//be [0x4df779]
    /usr/open64/lib//be [0x42b936]
    /usr/open64/lib//be [0x42bba0]
    /usr/open64/lib//be [0x42bc9d]
    /usr/open64/lib//be [0x42c056]
    /usr/open64/lib//be [0x42c23d]
    /usr/open64/lib//be [0x42165a]
    /usr/open64/lib//be [0x4781fd]
    /usr/open64/lib//be [0x4043a2]
    /usr/open64/lib//be [0x40502e]
    /usr/open64/lib//be [0x406081]
    /usr/open64/lib//be [0x4073ad]
    /lib64/libc.so.6(__libc_start_main+0xe6) [0x2ac946293586]
    /usr/open64/lib//be [0x4037ea]
nvopencc INTERNAL ERROR: /usr/open64/lib//be died due to signal 4
--------------------------------------------------------------------------------

system information:
Linux openSUSE-11.1 x86_64 kernel 2.6.27.19-3.2-default

nvcc version:
Built on Thu_Mar__5_04:25:57_PST_2009
Cuda compilation tools, release 2.2, V0.2.1221

gcc version:
gcc (SUSE Linux) 4.3.2 [gcc-4_3-branch revision 141291]

*/

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
  float4 v = make_float4(1, 1, 1, 1);
  *p = m * m * v;
}

int
main()
{
}
