#include "cudatemplates/hostmemoryheap.hpp"
#include "cudatemplates/devicememorylinear.hpp"
#include "cudatemplates/copy.hpp"


using namespace Cuda;

int main() {  
  size_t my_size = 256;
  int* input = new int[my_size];
  
  for (size_t i = 0; i < my_size; i++)
	input[i] = i;  
  
#if 0
  // this creates a function pointer of type "Cuda::HostMemoryHeap<int, 1u> (*)(Cuda::Size<1u>)":
  HostMemoryHeap<int,1> host(Size<1>(my_size));
#else
  // this creates an object of type "Cuda::HostMemoryHeap<int, 1u>":
  Size<1> size(my_size);
  HostMemoryHeap<int,1> host(size);
#endif

  for (size_t i = 0; i < my_size; i++)
    host.getBuffer()[i] = input[i];  

  DeviceMemoryLinear<int, 1>* device = new DeviceMemoryLinear<int, 1>(host);

  copy(host, *device);

  delete device;
  delete[] input;
  
  return 0;
}
