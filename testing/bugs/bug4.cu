/*

Compiling this file with "nvcc --keep bug4.cu" gives the following error message:

--------------------------------------------------------------------------------
In file included from bug4.cu:39:
bug4.cudafe1.stub.c: In function ‘void __sti____cudaRegisterAll_12_bug4_cpp1_ii_main()’:
bug4.cudafe1.stub.c:23: error: insufficient contextual information to determine type
--------------------------------------------------------------------------------

The error does not appear when "bool" is replaced by "int".

system information:
Linux openSUSE-11.1 x86_64 kernel 2.6.27.19-3.2-default

nvcc version:
Built on Thu_Mar__5_04:25:57_PST_2009
Cuda compilation tools, release 2.2, V0.2.1221

gcc version:
gcc (SUSE Linux) 4.3.2 [gcc-4_3-branch revision 141291]

*/


template<class T>
__global__ void kernel(const T *p, bool x)
{
}

int
main()
{
  dim3 dimGrid, dimBlock;
  kernel<<<dimGrid, dimBlock>>>((float *)0, true);
}
