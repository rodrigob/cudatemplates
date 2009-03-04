#include <assert.h>

#include <cuda.h>


#define SIZE 1024


__constant__ float data[SIZE];


int
main()
{
  size_t symsize1, symsize2;
  cudaGetSymbolSize(&symsize1, data);
  cudaGetSymbolSize(&symsize2, "data");
  assert(symsize1 == SIZE * sizeof(data[0]));
  assert(symsize2 == SIZE * sizeof(data[0]));
}
