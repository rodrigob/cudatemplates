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

#ifndef CUDA_HOSTMEMORYREFERENCE_H
#define CUDA_HOSTMEMORYREFERENCE_H


#include <cudatemplates/hostmemory.hpp>


#define CUDA_INIT_POINTER(...) __VA_ARGS__,


namespace Cuda {

/**
   Reference to existing buffer in CPU main memory.
   This class can be used to apply the CUDA Toolkit methods to memory regions
   managed by other libraries.
*/
template <class Type, unsigned Dim>
class HostMemoryReference: public HostMemory<Type, Dim>
{
public:
  /**
     Constructor.
     @param requested layout
     @param _buffer pointer to CPU memory
  */
  inline HostMemoryReference(const Layout<Type, Dim> &layout, Type *_buffer):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    HostMemory<Type, Dim>(layout)
  {
    this->buffer = _buffer;
  }

protected:
  /**
     Default constructor.
     This is only for subclasses which know how to correctly set up the data
     pointer.
  */
  inline HostMemoryReference() {}

};

CUDA_SPECIALIZE_DIM(HostMemoryReference);

}


#undef CUDA_INIT_POINTER


#endif
