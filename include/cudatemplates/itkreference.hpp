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

#ifndef CUDA_ITK_REFERENCE_H
#define CUDA_ITK_REFERENCE_H


#include <itkImage.h>

#include <cudatemplates/hostmemoryreference.hpp>


namespace Cuda {

/**
   Reference to existing ITK image.
*/
template <class Type, unsigned Dim>
class ItkReference:
    virtual public Layout<Type, Dim>,
    virtual public Pointer<Type, Dim>,
    public HostMemoryReference<Type, Dim>
{
public:
  typedef typename itk::Image<Type, Dim> ImageType;

  /**
     Constructor.
     Creates an ItkReference by using an existing, already allocated ITK image!
     @param image ITK image to reference
  */
  inline ItkReference(typename ImageType::Pointer image):
    Layout<Type, Dim>(),
    Pointer<Type, Dim>(),
    HostMemoryReference<Type, Dim>(),
    min_orig_gray_value(0),
    max_orig_gray_value(0)
  {
    setImage(image);
  }

  /**
     Constructor.
     Creates an ItkReference from an ITK image that has been constructed (Image::New() has been called)
     but has not been allocated with memory! This constructor allocates memory for the ITK image of the
     given size internally and also sets the spacing accordingly.
     @param _size size of image
     @param _spacing spacing of image
     @param image ITK image to reference
  */
  inline ItkReference(const Size<Dim> &_size, const Spacing<Dim> &_spacing, typename ImageType::Pointer image):
    Layout<Type, Dim>(),
    Pointer<Type, Dim>(),
    HostMemoryReference<Type, Dim>(),
    min_orig_gray_value(0),
    max_orig_gray_value(0)
  {
    typename ImageType::IndexType istart;
    typename ImageType::SizeType isize;
    typename ImageType::SpacingType ispacing;

    for(unsigned i = Dim; i--;) {
      istart[i] = 0;
      isize[i] = _size[i];
      ispacing[i] = _spacing[i];
    }

    typename ImageType::RegionType region;
    region.SetSize(isize);
    region.SetIndex(istart);
    image->SetRegions(region);
    image->SetSpacing(ispacing);
    image->Allocate();
    setImage(image);
  }

  /**
    Access to the ITK image pointer, useful for writing an image.
  */
  typename ImageType::Pointer getITKImagePointer()
  {
     return image_ptr;
  }

  /**
    Minimum gray value in original ITK data.
    Used for convenience to store original grayvalue range to be used during denormalization.
  */
  Type min_orig_gray_value;


  /**
    Maximum gray value in original ITK data.
    Used for convenience to store original grayvalue range to be used during denormalization.
  */
  Type max_orig_gray_value;

private:
  /**
     Extract image information from ITK image.
     @param image ITK image
  */
  void setImage(typename ImageType::Pointer image)
  {
    image_ptr = image;
    typename ImageType::SizeType isize = image->GetLargestPossibleRegion().GetSize();
    typename ImageType::SpacingType ispacing = image->GetSpacing();

    for(unsigned i = Dim; i--;) {
      this->size[i] = isize[i];
      this->spacing[i] = ispacing[i];
    }

    this->buffer = image->GetBufferPointer();
    this->setPitch(0);
  }

  /**
     Pointer to ITK image.
     This pointer is not used in the ItkReference class, but it increases the
     reference count of the ITK image and prevents it from being deleted during
     the lifetime of the ItkReference instance.
  */
  typename ImageType::Pointer image_ptr;
};

}  // namespace Cuda


#include "auto/specdim_itkreference.hpp"


#endif
