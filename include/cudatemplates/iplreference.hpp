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


namespace Cuda {

/**
   Reference to existing IPL image.
*/
template <class Type, unsigned Dim>
class IplReference:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public HostMemoryReference<Type, Dim>
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

  /**
     Constructor.
     This allocates memory for the IplImage of the given size.
     @param _size size of image
     @param _image pointer to IplImage to be referenced.
  */
  inline IplReference(const Size<Dim> &_size, IplImage *_image):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    HostMemoryReference<Type, Dim>()
  {
    setImage(_image);
  }

  /**
     Returns the IplImage pointer
  */
  inline IplImage* getIplImage()
    {
      return(image_ptr_);
    }

  /**
     Extract image information from IplImage. Also resets size, region_size and
     region_ofs to offer the possibility to update the buffer of an existing
     IplReference.
     @param image IplImage to be referenced.
  */
  void setImage(IplImage *image)
  {
    image_ptr_ = image;
    
    this->size = Size<2>(image->width, image->height);
    this->region_ofs = Size<2>(0,0);
    this->region_size = this->size;
    this->setPitch(image->widthStep);
    num_channels_ = image_ptr_->nChannels;

    // buffer is always interpreted as given template type -> TODO convert function?
    this->buffer = (Type*)image->imageData;
  }

protected:
  /**
     Pointer to IplImage.  This pointer is not used in the
     IplReference class, but it offers the possibility to handle smart
     pointers at some time. In future the IplReference should handle
     the memory of the referenced IplImage.
  */
  IplImage *image_ptr_;

  int num_channels_; /**< Number of channels that are used to save image data. */
};

}  // namespace Cuda


#include "auto/specdim_iplreference.hpp"


#endif
