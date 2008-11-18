/* 
  Cuda Templates.

  Copyright (C) 2008 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUFFTPP_H
#define CUFFTPP_H


#include <cassert>
#include <complex>

#include <cufft.h>

#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/error.hpp>


//!!! #define CUFFT_CHECK(call) { result_t_t err = call; if(err != CUFFT_SUCCESS) throw Cuda::Error(__FILE__, __LINE__, __PRETTY_FUNCTION__, (int)err, 0); }
#define CUFFT_CHECK(call) call


namespace Cuda {
namespace FFT {

typedef cufftReal real;
typedef std::complex<cufftReal> std_complex;
typedef cufftComplex complex;
typedef cufftType_t type_t;
typedef cufftResult_t result_t;


#define CUFFT_PLAN_SIZE1 size[0]
#define CUFFT_PLAN_SIZE2 size[0], size[1]
#define CUFFT_PLAN_SIZE3 size[0], size[1], size[2]


// TODO: exec() methods must check for contiguous memory!


#define CUFFT_PLAN_GENERIC(type, dim)					\
  private:								\
  cufftHandle plan;							\
  public:								\
  inline ~Plan()							\
  {									\
    CUFFT_CHECK(cufftDestroy(plan));					\
  }
  
#define CUFFT_PLAN_CONSTRUCTOR(type, dim)				\
  public:								\
  inline Plan(const Size<dim> &size)					\
  {									\
    CUFFT_CHECK(cufftPlan ## dim ## d					\
		(&plan, CUFFT_PLAN_SIZE ## dim, CUFFT_ ## type));	\
  }

#define CUFFT_PLAN_CONSTRUCTOR_BATCH(type, dim)				\
  private:								\
  int batch;								\
  public:								\
  inline Plan(const Size<dim> &size, int _batch = 1):			\
    batch(_batch)							\
  {									\
    CUFFT_CHECK(cufftPlan ## dim ##d					\
		(&plan, CUFFT_PLAN_SIZE ## dim,				\
		 CUFFT_ ## type, batch));				\
  }

#define CUFFT_PLAN_EXEC(type, dim, in, out)				\
  public:								\
  inline void exec(const DeviceMemory<in, dim> &idata,			\
		   DeviceMemory<out, dim> &odata)			\
  {									\
    CUFFT_CHECK(cufftExec ## type					\
		(plan, const_cast<in *>(idata.getBuffer()),		\
		 odata.getBuffer()));					\
  }
  
#define CUFFT_PLAN_EXEC_DIRECTION(type, dim, in, out)			\
  public:								\
  inline void exec(const DeviceMemory<in, dim> &idata,			\
		   DeviceMemory<out, dim> &odata, int dir)		\
  {									\
    CUFFT_CHECK(cufftExec ## type					\
		(plan, const_cast<in *>(idata.getBuffer()),		\
		 odata.getBuffer(), dir));				\
  }
  

template <class TypeIn, class TypeOut, unsigned Dim>
class Plan
{
};

template <>
class Plan<real, complex, 1>
{
  CUFFT_PLAN_GENERIC          (R2C, 1)
  CUFFT_PLAN_CONSTRUCTOR_BATCH(R2C, 1)
  CUFFT_PLAN_EXEC             (R2C, 1, real, complex)
};

template <>
class Plan<complex, real, 1>
{
  CUFFT_PLAN_GENERIC          (C2R, 1)
  CUFFT_PLAN_CONSTRUCTOR_BATCH(C2R, 1)
  CUFFT_PLAN_EXEC             (C2R, 1, complex, real)
};

template <>
class Plan<complex, complex, 1>
{
  CUFFT_PLAN_GENERIC          (C2C, 1)
  CUFFT_PLAN_CONSTRUCTOR_BATCH(C2C, 1)
  CUFFT_PLAN_EXEC_DIRECTION   (C2C, 1, complex, complex)
};

template <>
class Plan<real, complex, 2>
{
  CUFFT_PLAN_GENERIC          (R2C, 2)
  CUFFT_PLAN_CONSTRUCTOR      (R2C, 2)
  CUFFT_PLAN_EXEC             (R2C, 2, real, complex)
};

template <>
class Plan<complex, real, 2>
{
  CUFFT_PLAN_GENERIC          (C2R, 2)
  CUFFT_PLAN_CONSTRUCTOR      (C2R, 2)
  CUFFT_PLAN_EXEC             (C2R, 2, complex, real)
};

template <>
class Plan<complex, complex, 2>
{
  CUFFT_PLAN_GENERIC          (C2C, 2)
  CUFFT_PLAN_CONSTRUCTOR      (C2C, 2)
  CUFFT_PLAN_EXEC_DIRECTION   (C2C, 2, complex, complex)
};

template <>
class Plan<real, complex, 3>
{
  CUFFT_PLAN_GENERIC          (R2C, 3)
  CUFFT_PLAN_CONSTRUCTOR      (R2C, 3)
  CUFFT_PLAN_EXEC             (R2C, 3, real, complex)
};

template <>
class Plan<complex, real, 3>
{
  CUFFT_PLAN_GENERIC          (C2R, 3)
  CUFFT_PLAN_CONSTRUCTOR      (C2R, 3)
  CUFFT_PLAN_EXEC             (C2R, 3, complex, real)
};

template <>
class Plan<complex, complex, 3>
{
  CUFFT_PLAN_GENERIC          (C2C, 3)
  CUFFT_PLAN_CONSTRUCTOR      (C2C, 3)
  CUFFT_PLAN_EXEC_DIRECTION   (C2C, 3, complex, complex)
};


#undef CUFFT_PLAN_SIZE1
#undef CUFFT_PLAN_SIZE2
#undef CUFFT_PLAN_SIZE3
#undef CUFFT_PLAN_GENERIC
#undef CUFFT_PLAN_CONSTRUCTOR
#undef CUFFT_PLAN_CONSTRUCTOR_BATCH
#undef CUFFT_PLAN_EXEC
#undef CUFFT_PLAN_EXEC_DIRECTION

}  // namespace FFT
}  // namespace Cuda


#undef CUFFT_CHECK

#endif
