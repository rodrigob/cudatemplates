/*
  NOTE: THIS FILE HAS BEEN CREATED AUTOMATICALLY,
  ANY CHANGES WILL BE OVERWRITTEN WITHOUT NOTICE!
*/

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

#ifndef CUFFT_REAL_COMPLEX_1D_H
#define CUFFT_REAL_COMPLEX_1D_H


#include <cudatemplates/cufft_common.hpp>
#include <cudatemplates/devicememory.hpp>


namespace Cuda {
namespace FFT {

/**
   Plan for real-to-complex FFT.
*/
template <>
class Plan<real, complex, 1>
{
public:
  enum { Dim = 1 };

  /**
     Constructor.
     The constructor creates a CUFFT plan with the given size.
     @param size requested size of CUFFT plan
     @param batch number of 1D transforms
  */
  inline Plan(const Size<1> &size, int batch = 1)
  {
    CUFFT_CHECK(cufftPlan1d(&plan, size[0], CUFFT_R2C, batch));
  }

  /**
     Constructor.
     The constructor creates a CUFFT plan with the given size.
     @param size0 requested size of CUFFT plan
     @param batch number of 1D transforms
  */
  inline Plan(size_t size0, int batch = 1)
  {
    CUFFT_CHECK(cufftPlan1d(&plan, size0, CUFFT_R2C, batch));
  }

  /**
     Destructor.
     The destructor destroys the CUFFT plan.
  */
  inline ~Plan()
  {
    CUFFT_CHECK(cufftDestroy(plan));
  }

  /**
     Executes the CUFFT plan.
     This method executes the CUFFT real-to-complex transform plan. CUFFT
     uses as input data the GPU memory specified by the idata parameter. The
     Fourier coefficients are stored in the odata array. If idata and odata
     refer to the same memory location, this method does an in‐place transform.
     @param idata input data
     @param odata output data
     
  */
  inline void exec(const DeviceMemory<real, 1> &idata, DeviceMemory<complex, 1> &odata)
  {
    if((Dim > 1) && !(idata.contiguous() && odata.contiguous()))
      CUDA_ERROR("CUFFT can only be used for contiguous memory (i.e., no padding between rows)");

    CUFFT_CHECK(cufftExecR2C(plan, const_cast<real *>(idata.getBuffer()), odata.getBuffer()));
  }
  
private:
  cufftHandle plan;
};

}  // namespace FFT
}  // namespace Cuda


#endif
