#ifndef CUDA_GCC43_COMPAT_H
#define CUDA_GCC43_COMPAT_H


/**
   CUDA-2.0 and gcc-4.3 are not compatible with respect to some C++ features.
   This file tries to detect the relevant cases and provides some macros as a
   workaround to allow the use of CUDA with recent versions of gcc.
*/


#include <cuda.h>


#if (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 3)) && defined(__CUDACC__) && (CUDA_VERSION <= 2000)

#include <stdarg.h>

#undef __GXX_WEAK__
#define __GXX_WEAK__ 0
#define __builtin_memchr memchr
#define __builtin_va_list va_list
#define __builtin_vsnprintf vsnprintf
#define __is_empty(x) (sizeof(x) == 0)

/*
  The following definition will definitely break some code.
  If it breaks *your* code, feel free to improve, and read:
  http://www.fnal.gov/docs/working-groups/fpcltf/Pkg/ISOcxx/doc/POD.html
  http://www.parashift.com/c++-faq-lite/intrinsic-types.html#faq-26.7
  http://en.wikipedia.org/wiki/Plain_Old_Data_Structures
*/
#define __is_pod(x) false

#endif


#endif
