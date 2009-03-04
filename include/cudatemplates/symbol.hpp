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

#ifndef CUDA_SYMBOL_H
#define CUDA_SYMBOL_H


#include <cudatemplates/error.hpp>
#include <cudatemplates/layout.hpp>


namespace Cuda {

/**
   Representation of CUDA symbol.
   CUDA symbols are used to access data residing in global or constant memory
   space declared as global variables in CUDA code (__constant__ or __device__
   qualifiers at file scope). The CUDA templates implementation can be used for
   C-style arrays like this:

\code
__constant__ float data[1024];
Cuda::Symbol<float, 1> symbol(Cuda::Size<1>(1024), data);
\endcode

   Size and data type of the array and the symbol must match.
*/
template <class Type, unsigned Dim>
class Symbol: virtual public Layout<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Symbol():
    symbol(0)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of symbol
     @param _symbol symbol in global or constant memory space
     The constructor fails if the size of the symbol as reported by CUDA
     doesn't match the requested size passed as an argument.
  */
  inline Symbol(const Size<Dim> &_size, Type *_symbol):
    Layout<Type, Dim>(_size),
    symbol(_symbol)
  {
    checkSize();
  }

  /**
     Constructor.
     @param layout requested layout of symbol
     @param _symbol symbol in global or constant memory space
     The constructor fails if the size of the symbol as reported by CUDA
     doesn't match the requested layout passed as an argument.
  */
  inline Symbol(const Layout<Type, Dim> &layout, Type *_symbol):
    Layout<Type, Dim>(layout),
    symbol(_symbol)
  {
    checkSize();
  }

  /**
     Get the symbol address.
     This is not the address of the data in device memory, but a reference to
     the data structure in host memory representing the symbol. You will never
     need this unless you directly call CUDA functions related to symbols.
     @return reference to symbol data structure in host memory
  */
  inline const Type &getSymbol() const { return *symbol; }

  /**
     Get the address of the data in global memory space.
     This creates a runtime error when called for data residing in constant
     memory space.
     @return address of data in global memory space
  */
  const Type *getBuffer() const
  {
    const Type *buffer;
    CUDA_CHECK(cudaGetSymbolAddress((void **)&buffer, (const char *)symbol));
    return buffer;
  }

  /**
     Get the address of the data in global memory space.
     This creates a runtime error when called for data residing in constant
     memory space.
     @return address of data in global memory space
  */
  inline Type *getBuffer() { return const_cast<Type *>(const_cast<const Symbol<Type, Dim> *>(this)->getBuffer()); }

private:
  Type *symbol;

  void
  checkSize()
  {
    size_t symsize = 0;
    CUDA_CHECK(cudaGetSymbolSize(&symsize, (const char *)symbol));

    if(symsize != this->getBytes())
      CUDA_ERROR("symbol size mismatch");
  }
};

}  // namespace Cuda


#include "auto/specdim_symbol.hpp"


#endif
