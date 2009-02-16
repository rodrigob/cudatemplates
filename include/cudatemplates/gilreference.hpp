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
    template <class T> struct pixel {};

    // supported types:

#define CUDA_GIL_PIXEL(T, c, l) template <> struct pixel<T> { typedef c channel; typedef boost::gil::l ## _layout_t layout; }

    CUDA_GIL_PIXEL(char, char, gray);
    CUDA_GIL_PIXEL(struct char1, char, gray);
    CUDA_GIL_PIXEL(struct char3, char, rgb);
    CUDA_GIL_PIXEL(struct char4, char, rgba);

    CUDA_GIL_PIXEL(unsigned char, unsigned char, gray);
    CUDA_GIL_PIXEL(struct uchar1, unsigned char, gray);
    CUDA_GIL_PIXEL(struct uchar3, unsigned char, rgb);
    CUDA_GIL_PIXEL(struct uchar4, unsigned char, rgba);

    CUDA_GIL_PIXEL(short, short, gray);
    CUDA_GIL_PIXEL(struct short1, short, gray);
    CUDA_GIL_PIXEL(struct short3, short, rgb);
    CUDA_GIL_PIXEL(struct short4, short, rgba);

    CUDA_GIL_PIXEL(unsigned short, unsigned short, gray);
    CUDA_GIL_PIXEL(struct ushort1, unsigned short, gray);
    CUDA_GIL_PIXEL(struct ushort3, unsigned short, rgb);
    CUDA_GIL_PIXEL(struct ushort4, unsigned short, rgba);

    CUDA_GIL_PIXEL(int, int, gray);
    CUDA_GIL_PIXEL(struct int1, int, gray);
    CUDA_GIL_PIXEL(struct int3, int, rgb);
    CUDA_GIL_PIXEL(struct int4, int, rgba);

    CUDA_GIL_PIXEL(unsigned int, unsigned int, gray);
    CUDA_GIL_PIXEL(struct uint1, unsigned int, gray);
    CUDA_GIL_PIXEL(struct uint3, unsigned int, rgb);
    CUDA_GIL_PIXEL(struct uint4, unsigned int, rgba);

    CUDA_GIL_PIXEL(float, boost::gil::bits32f, gray);
    CUDA_GIL_PIXEL(struct float1, boost::gil::bits32f, gray);
    CUDA_GIL_PIXEL(struct float3, boost::gil::bits32f, rgb);
    CUDA_GIL_PIXEL(struct float4, boost::gil::bits32f, rgba);

#undef CUDA_GIL_PIXEL

  }

/**
   Reference to existing boost::gil image.
*/
template <class Type>
class GilReference2D:
    virtual public Layout<Type, 2>,
    virtual public Pointer<Type, 2>,
    public HostMemoryReference2D<Type>
{
public:
  typedef boost::gil::image<boost::gil::pixel<typename Cuda::gil::pixel<Type>::channel,
					      typename Cuda::gil::pixel<Type>::layout>,
			    false> gil_image_t;

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
    this->region_ofs[0] = 0;
    this->region_ofs[1] = 0;
    this->region_size = this->size;
    
    this->spacing[0] = this->spacing[1] = 1;
    this->buffer = reinterpret_cast<Type *>(&boost::gil::view(image).pixels()[0]);
    this->setPitch(0);  // TODO: get this from gil::image!
  }
};

}  // namespace Cuda


// #include "auto/specdim_gilreference.hpp"


#endif
