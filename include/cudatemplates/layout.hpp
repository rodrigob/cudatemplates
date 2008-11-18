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

#ifndef CUDA_LAYOUT_H
#define CUDA_LAYOUT_H


#include <cassert>

#include <cudatemplates/size.hpp>


/*
  Default constructors are convenient for application programmers using this
  library, but due to the use of virtual base classes the library developer
  can easily pick the wrong constructor, therefore this flag is introduced to
  disable default constructors and get compiler error messages where the
  default constructor would have been used incorrectly.
  This is for testing purposes only, do *NOT* set this flag in application
  code!
*/
// #define CUDA_NO_DEFAULT_CONSTRUCTORS


/**
   This namespace contains all classes and functions of the CUDA templates.
*/
namespace Cuda {

/**
   Description of memory layout of multidimensional data.
   To achieve optimal memory bandwidth in CUDA, it is crucial to follow some
   memory layout rules. The purpose of this class is to hold the layout
   information in addition to the actual size of the data. Moreover, the Layout
   class contains fields for specifying geometric properties in each dimension
   (see Layout::spacing) and a region of interest (see Layout::region_ofs and
   see Layout::region_size). These fields are often used in application code
   and therefore provided for convenience, but are not handled by the Layout
   class itself.
*/
template <class _Type, unsigned _Dim>
class Layout
{
  CUDA_STATIC_ASSERT(_Dim > 0);

public:
  typedef _Type Type;
  enum { Dim = _Dim };
  typedef Layout<_Type, _Dim> layout_t;

#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  Layout() :
    xsize(0),
    ysize(0)
  {
    for(int i = Dim; i--;) {
      size[i] = 0;
      spacing[i] = 1;
      stride[i] = 0;
    }
  }
#endif

  /**
     Constructor.
     @param _size requested size (may be padded by subclasses to fulfill
     alignment requirements).
  */
  Layout(const Size<Dim> &_size):
    region_size(_size),
    xsize(0),
    ysize(0)
  {
    for(int i = Dim; i--;)
    {
      spacing[i] = 1;
      region_ofs[i] = 0;
    }

    setSize(_size);
  }

  /**
     Get total size in bytes.
     @return number of totally allocated bytes (including any padding)
  */
  inline size_t getBytes() const { return getSize() * sizeof(Type); }

  /**
     Get offset for given index.
     @param index index for which to compute offset
     @return offset (in elements) for given index
  */
  size_t getOffset(const SizeBase<Dim> &index) const
  {
    size_t o = index[0];

    for(size_t i = Dim; --i;)
      o += index[i] * stride[i - 1];

    return o;
  }

  /**
     Determine if layout is contiguous.
  */
  bool contiguous() const
  {
    Layout<Type, Dim> x(*this);
    x.setPitch(0);
    return x.stride == stride;
  }

  /**
     Get pitch.
     @return number of bytes in a row (including any padding)
  */
  inline size_t getPitch() const { return stride[0] * sizeof(Type); }

  /**
     Get total number of elements.
     @return number of totally allocated elements (including any padding)
  */
  inline size_t getSize() const { return stride[Dim - 1]; }

  /**
     Set layout.
     @param layout new layout
  */
  inline void setLayout(const Layout<Type, Dim> &layout) { *this = layout; }

  /**
     Set pitch.
     Computes the step size in each dimension for later offset calculations.
     The image size must be set before calling this method.
     @param pitch new pitch
   */
  void setPitch(size_t pitch)
  {
    if(pitch == 0)
      pitch = size[0] * sizeof(Type);

    assert(pitch % sizeof(Type) == 0);
    stride[0] = pitch / sizeof(Type);
  
    for(unsigned i = 1; i < Dim; ++i)
      stride[i] = stride[i - 1] * size[i];
  }

  /**
     Set size.
     This also removed any padding.
     @param _size new size
  */
  void setSize(const Size<Dim> &_size)
  {
    for(size_t i = Dim; i--;)
      size[i] = _size[i];
    
    setPitch(0);  // no padding by default
  }

  void setRegion(const Size<Dim> &_ofs, const Size<Dim> &_size)
  {
    region_ofs = _ofs;
    region_size = _size;
  }

  /**
     Size of the layout in each dimension (in elements).
  */
  Size<Dim> size;

  /**
     Offset of region of interesst
  */
  Size<Dim> region_ofs;

  /**
     Size of region of interest
  */
  Size<Dim> region_size;

  /**
     Step size of the layout in each dimension (in elements).
  */
  Size<Dim> stride;

  /**
     Spacing in each dimension.
     This is currently unused.
  */
  float spacing[Dim];

  /**
     The xsize field specifies the logical width of each row in bytes.
     Currently only used for CUDA 3d alloc and memset
  */
  size_t xsize;

  /**
     The ysize field specifies the height of each 2D slice in rows.
     Currently only used for CUDA 3d alloc and memset
  */
  size_t ysize;
  
};

}


#endif
