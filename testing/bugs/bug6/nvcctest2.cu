__global__ void kernel()
{
}

template <class T>
void launch()
{
  kernel<<<dim3(1), dim3(1)>>>();
}
