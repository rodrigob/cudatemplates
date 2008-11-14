/**

\mainpage CUDA templates -  a C++ interface to the NVIDIA CUDA Toolkit

\section intro Introduction

"CUDA Templates" is a collection of C++ template classes and functions
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

\section example Example

Here is a small example which demonstrates the basic features of CUDA templates.
The following code uses "pure" CUDA:

\code
// to be written...
\endcode

The CUDA templates equivalent of this code is given below:

\code
// to be written...
\endcode

*/
