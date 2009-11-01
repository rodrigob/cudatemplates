/**

\mainpage CUDA templates -  a C++ interface to the NVIDIA CUDA Toolkit

\section intro Introduction

The "CUDA templates" are a collection of C++ template classes and functions
which provide a consistent interface to
<a href="http://www.nvidia.com">NVIDIA</a>'s "Compute Unified Device Architecture"
(<a href="http://www.nvidia.com/object/cuda_home.html">CUDA</a>),
hiding much of the complexity of the underlying CUDA functions from the programmer.

\subsection main_features Main features
The main features of the CUDA templates are:
<ul>
<li>A consistent API is provided to allocate data of various data types and dimensions
across all storage types available in CUDA. This includes host memory (CPU),
linear device memory (GPU), and texture memory (GPU).<!-- See ... for more details. -->
<li>A consistent API is provided to copy data between any two blocks of memory
of the same data type, dimension, and size.
By means of function overloading, the compiler automatically selects the appropriate
copy (template) function and calls the corresponding low-level copy function.<!-- See ... for more details. -->
<li>Errors reported by the underlying libraries (CUDA, OpenGL)
are transformed into C++ exceptions.
<li>Interfaces to several image libraries (such as
<a href="http://www.itk.org">Insight Toolkit</a>,
<a href="http://www.boost.org/doc/libs/1_37_0/libs/gil/doc/index.html">gil</a>, and
<a href="http://sourceforge.net/projects/opencvlibrary">OpenCV</a>)
are provided for convenience.
</ul>

\subsection requirements Requirements
The following components must be installed on your machine before you can use the CUDA templates:
<ul>
<li>CUDA Driver, Toolkit, and SDK, version 2.3 or newer is recommended
(<a href="http://www.nvidia.com/object/cuda_get.html">Download</a>)
<li>FindCUDA cmake modules
(<a href="https://gforge.sci.utah.edu/gf/project/findcuda">Download</a>)
</ul>

\section example Example

Here is a small
<a href="https://cudatemplates.svn.sourceforge.net/svnroot/cudatemplates/trunk/testing/demo.cpp">example</a>
which demonstrates the basic features of the CUDA templates.
It allocates two three-dimensional arrays in host memory and one corresponding array in device memory.
Data is copied from the first host array to the device and then back to the second host array.
Finally the allocated resources are freed. Full error checking is provided.

The following code uses the CUDA runtime functions:

\code
// allocate host memory:
float *mem_host1 = (float *)malloc(sizeof(float) * SIZE[0] * SIZE[1] * SIZE[2]);
float *mem_host2 = (float *)malloc(sizeof(float) * SIZE[0] * SIZE[1] * SIZE[2]);

if((mem_host1 == 0) || (mem_host2 == 0)) {
  cerr << "out of memory\n";
  exit(1);
}

// init host memory:
init(mem_host1);

// allocate device memory:
cudaExtent extent;
extent.width = SIZE[0];
extent.height = SIZE[1];
extent.depth = SIZE[2];
cudaPitchedPtr mem_device;
CUDA_CHECK(cudaMalloc3D(&mem_device, extent));

// copy from host memory to device memory:
cudaMemcpy3DParms p = { 0 };
p.srcPtr.ptr = mem_host1;
p.srcPtr.pitch = SIZE[0] * sizeof(float);
p.srcPtr.xsize = SIZE[0];
p.srcPtr.ysize = SIZE[1];
p.dstPtr.ptr = mem_device.ptr;
p.dstPtr.pitch = mem_device.pitch;
p.dstPtr.xsize = SIZE[0];
p.dstPtr.ysize = SIZE[1];
p.extent.width = SIZE[0] * sizeof(float);
p.extent.height = SIZE[1];
p.extent.depth = SIZE[2];
p.kind = cudaMemcpyHostToDevice;
CUDA_CHECK(cudaMemcpy3D(&p));

// copy from device memory to host memory:
p.srcPtr.ptr = mem_device.ptr;
p.srcPtr.pitch = mem_device.pitch;
p.dstPtr.ptr = mem_host2;
p.dstPtr.pitch = SIZE[0] * sizeof(float);
p.kind = cudaMemcpyDeviceToHost;
CUDA_CHECK(cudaMemcpy3D(&p));

// verify host memory:
verify(mem_host2);

// free memory:
CUDA_CHECK(cudaFree(mem_device.ptr));
free(mem_host2);
free(mem_host1);
\endcode

The CUDA templates equivalent of this code is given below.
Every operation is performed by a single statement.
Resource deallocation is implicit.

\code
try {
  // allocate host memory:
  Cuda::HostMemoryHeap3D<float> mem_host1(SIZE[0], SIZE[1], SIZE[2]);
  Cuda::HostMemoryHeap3D<float> mem_host2(SIZE[0], SIZE[1], SIZE[2]);

  // init host memory:
  init(mem_host1.getBuffer());

  // allocate device memory:
  Cuda::DeviceMemoryPitched3D<float> mem_device(SIZE[0], SIZE[1], SIZE[2]);

  // copy from host memory to device memory:
  copy(mem_device, mem_host1);

  // copy from device memory to host memory:
  copy(mem_host2, mem_device);

  // verify host memory:
  verify(mem_host2.getBuffer());
}
catch(const exception &e) {
  cerr << e.what();
}
\endcode

*/
