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
