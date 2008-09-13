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

#ifndef CUDA_IPL_REFERENCE_H
#define CUDA_IPL_REFERENCE_H


#include "cv.h"

#include <cudatemplates/hostmemoryreference.hpp>


#define CUDA_INIT_POINTER(...) __VA_ARGS__,


namespace Cuda {

/**
   Reference to existing IPL image.
*/
template <class Type, unsigned Dim>
class IplReference: public HostMemoryReference<Type, Dim>
{
public:
  /**
     Constructor.
     @param image IPL image to reference
  */
  inline IplReference(IplImage *image):
    Layout<Type, Dim>(),
    Pointer<Type, Dim>(),
    HostMemoryReference<Type, Dim>()
  {
    setImage(image);
  }

private:
  /**
     Extract image information from IPL image.
     @param image IPL image
  */
  void setImage(IplImage *image)
  {
    image_ptr = image;

    this->size[0] = image->width;
    this->size[1] = image->height;
    this->setPitch(image->widthStep);

    // DO NOT set spacing with image->align -- this will cause a segfault!
    // this->spacing[1] = 
    // this->spacing[2] =

    unsigned char *tmp = (unsigned char*)image->imageData;
    this->buffer = reinterpret_cast<Type*>(tmp);
  }

  /**
     Pointer to IPL image.  This pointer is not used in the
     IplReference class, but it offers the possibility to handle smart
     pointers at some time. In future the IplReference should handle
     the memory of the referenced IplImage.
  */
  IplImage *image_ptr;
};

CUDA_SPECIALIZE_DIM(IplReference);

}  // namespace Cuda


#undef CUDA_INIT_POINTER


#endif
