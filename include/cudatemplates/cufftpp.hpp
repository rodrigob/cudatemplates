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
#include <stdexcept>

#include <cufft.h>

#include <cudatemplates/devicememorylinear.hpp>


#define CUFFT_CHECK(cmd) { result_t res = cmd; if(res != CUFFT_SUCCESS) throw Error(res); }


namespace Cuda {
namespace FFT {

typedef cufftReal real;
typedef std::complex<real> complex;
typedef cufftType_t type_t;
typedef cufftResult_t result_t;

class Error: public std::exception
{
  result_t result;

public:
  Error(result_t r): result(r) {}

  const char *what() const throw() {
    static char msg[100];
    sprintf(msg, "CUDA FFT error #%d", result);
    return msg;
  }
};

class Plan
{
protected:
  cufftHandle plan;

  inline ~Plan() {
    assert(sizeof(complex) == 2 * sizeof(real));
    CUFFT_CHECK(cufftDestroy(plan));
  }

public:
  inline void exec(const complex *idata, complex *odata, int direction) {
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *)idata, (cufftComplex *)odata, direction));
  }

  inline void exec(const real *idata, complex *odata) {
    CUFFT_CHECK(cufftExecR2C(plan, (cufftReal *)idata, (cufftComplex *)odata));
  }

  inline void exec(const complex *idata, real *odata) {
    CUFFT_CHECK(cufftExecC2R(plan, (cufftComplex *)idata, (cufftReal *)odata));
  }
};

class Plan1d: public Plan
{
public:
  inline Plan1d(int nx, type_t type, int batch = 1) {
    CUFFT_CHECK(cufftPlan1d(&plan, nx, type, batch));
  }

  inline void exec(const DeviceMemoryLinear<complex, 1> &idata, DeviceMemoryLinear<complex, 1> &odata, int direction) {
    // check size!!!
    Plan::exec(idata.getBuffer(), odata.getBuffer(), direction);
  }

  inline void exec(const DeviceMemoryLinear<real, 1> &idata, DeviceMemoryLinear<complex, 1> &odata) {
    // check size!!!
    Plan::exec(idata.getBuffer(), odata.getBuffer());
  }

  inline void exec(const DeviceMemoryLinear<complex, 1> &idata, DeviceMemoryLinear<real, 1> &odata) {
    // check size!!!
    Plan::exec(idata.getBuffer(), odata.getBuffer());
  }
};

class Plan2d: public Plan
{
public:
  inline Plan2d(int nx, int ny, type_t type) {
    CUFFT_CHECK(cufftPlan2d(&plan, nx, ny, type));
  }
};

class Plan3d: public Plan
{
public:
  inline Plan3d(int nx, int ny, int nz, type_t type) {
    CUFFT_CHECK(cufftPlan3d(&plan, nx, ny, nz, type));
  }
};

}  // namespace FFT
}  // namespace Cuda


#undef CUFFT_CHECK

#endif
