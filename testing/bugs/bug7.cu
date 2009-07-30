/*
  compiling this program with "nvcc -c bug7.cu" gives the following message:

  ### Assertion failure at line 123 of ../../be/cg/NVISA/expand.cxx:
  ### Compiler Error in file /tmp/tmpxft_000006e2_00000000-7_bug7.cpp3.i during Code_Expansion phase:
  ### unexpected mtype
  nvopencc INTERNAL ERROR: /usr/open64/lib//be returned non-zero status 1

  OS: Linux openSUSE-11.1 x86_64
  CUDA toolkit: 2.3
  gcc-4.3.2
*/

__global__ void kernel()
{
  float3 x0;
  float3 array[8];
  array[0] = x0;
  int i = 0;

  for(;;) {
    if(array[i].x < 0) {
      i++;
      continue;
    }

    array[i] = x0;
  }
}
