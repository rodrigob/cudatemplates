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

#ifndef CUDA_GIL_REFERENCE_H
#define CUDA_GIL_REFERENCE_H


#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

#include <cudatemplates/hostmemoryreference.hpp>


namespace Cuda {

  namespace gil {
    // pass types through to boost::gil except float:
    template <class src> struct typeconv { typedef src dst; };
    template <> struct typeconv<float> { typedef boost::gil::bits32f dst; };
  }

/**
   Reference to existing boost::gil image.
*/
template <class Type>
class GilReference2D: public HostMemoryReference2D<Type>
{
public:
  typedef boost::gil::image<boost::gil::pixel<typename Cuda::gil::typeconv<Type>::dst, boost::gil::gray_layout_t>, false> gil_image_t;

  /**
     Constructor.
     @param image boost::gil image to reference
  */
  inline GilReference2D(gil_image_t &image):
    Layout<Type, 2>(),
    Pointer<Type, 2>(),
    HostMemoryReference2D<Type>()
  {
    setImage(image);
  }

  /**
     Constructor.
     This allocates memory for the boost::gil image of the given size.
     @param _size size of image
     @param image boost::gil image to reference
  */
  inline GilReference2D(const Size<2> &_size, gil_image_t &image):
    Layout<Type, 2>(),
    Pointer<Type, 2>(),
    HostMemoryReference2D<Type>()
  {
    image.recreate(_size[0], _size[1]);
    setImage(image);
  }

private:
  /**
     Extract image information from boost::gil image.
     @param image boost::gil image
  */
  void setImage(gil_image_t &image)
  {
    this->size[0] = image.width();
    this->size[1] = image.height();
    this->spacing[0] = this->spacing[1] = 1;
    this->buffer = reinterpret_cast<Type *>(&boost::gil::view(image).pixels()[0]);
    this->setPitch(0);  // TODO: get this from gil::image!
  }
};

  // #include "specializations/gilreference.hpp"

}  // namespace Cuda


#endif
