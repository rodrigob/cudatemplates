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

#ifndef CUDA_IPLREFERENCE1D_H
#define CUDA_IPLREFERENCE1D_H


#include <cudatemplates/iplreference.hpp>


namespace Cuda {

/**
   IplReference template specialized for 1 dimension(s).
*/
template <class Type>
class IplReference1D : public IplReference<Type, 1>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline IplReference1D()
  {
  }
#endif


  /**
     Constructor.
     @param _image pointer to IplImage to be referenced.
  */
  inline IplReference1D(IplImage *_image):
    Layout<Type, 1>(),
    Pointer<Type, 1>(),
    IplReference<Type, 1>(_image)
  {
  }

  /**
     Constructor.
     @param _size size of memory block.
     @param _image pointer to IplImage to be referenced.
  */
  inline IplReference1D(const Size<1> &_size, IplImage *_image):
    Layout<Type, 1>(_size),
    Pointer<Type, 1>(_size),
    IplReference<Type, 1>(_size, _image)
  {
  }
};

}  // namespace Cuda


#endif
