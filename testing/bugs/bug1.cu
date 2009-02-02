/*
  compiling this program with "nvcc bug1.cu" gives the following message:

  nvcc error   : 'cudafe' died due to signal 11 (Invalid memory reference)

  Distribution: openSUSE-11.0
  Architecture: x86_64
  Linux kernel: 2.6.25.20
  CUDA toolkit: 2.1
  NVIDIA driver: 180.22
*/

template <class T>
struct A
{
  struct B
  {
    T x;
  };
};

__global__ void
kernel(A<int>::B x)
{
}
