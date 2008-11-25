#include <cudatemplates/cuda_gcc43_compat.hpp>

#include <fstream>
#include <iostream>
#include <string>

using namespace std;


/*
  The purpose of this file is to test the CUDA/gcc-4.3 compatibility layer.
  It is processed by nvcc and should compile on all platforms supported by
  CUDA.
*/


int
main()
{
  ofstream s("out.txt");
  string msg = "compatibility test";
  s << msg << endl;
  return 0;
}
