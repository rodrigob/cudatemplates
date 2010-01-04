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

#ifndef CUDA_TEXTURE1D_H
#define CUDA_TEXTURE1D_H


#include <cudatemplates/opengl/texture.hpp>


namespace Cuda {
namespace OpenGL {

/**
   Texture template specialized for 1 dimension(s).
*/
template <class Type>
class Texture1D:
    virtual public Layout<Type, 1>,
    virtual public Pointer<Type, 1>,
    public Texture<Type, 1>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Texture1D()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
  */
  inline Texture1D(const Size<1> &_size):
    Layout<Type, 1>(_size),
    Pointer<Type, 1>(_size),
    Texture<Type, 1>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout of memory block.
  */
  inline Texture1D(const Layout<Type, 1> &layout):
    Layout<Type, 1>(layout),
    Pointer<Type, 1>(layout),
    Texture<Type, 1>(layout)
  {
  }

  /**
     Constructor.
  */
  inline Texture1D(size_t size0):
    Layout<Type, 1>(Size<1>(size0)),
    Pointer<Type, 1>(Size<1>(size0)),
    Texture<Type, 1>(Size<1>(size0))
  {
  }

  /**
     Copy constructor.
     @param x instance of Texture1D to be copied
  */
  inline Texture1D(const Texture1D<Type> &x):
    Layout<Type, 1>(x),
    Pointer<Type, 1>(x),
    Texture<Type, 1>(x)
  {
  }

  /**
     Constructor.
     Initialization from different type.
     @param x instance of different type to be copied
  */
  template<class Name>
    inline Texture1D(const Name &x):
    Layout<Type, 1>(x),
    Pointer<Type, 1>(x),
    Texture<Type, 1>(x)
  {
  }

  /**
     Constructor.
     Initialization of region from same or different type.
     @param x instance to be copied
     @param ofs offset of region
     @param size size of region
  */
  template<class Name>
    inline Texture1D(const Name &x, const Size<1> &ofs, const Size<1> &size):
    Layout<Type, 1>(size),
    Pointer<Type, 1>(size),
    Texture<Type, 1>(x, ofs, size)
  {
  }

  /**
     Constructor.
     Initialization of region from same or different type.
     @param x instance to be copied
     @param ofs0 offset of region
     @param size0 size of region
  */
  template<class Name>
    inline Texture1D(const Name &x, size_t ofs0, size_t size0):
    Layout<Type, 1>(Size<1>(size0)),
    Pointer<Type, 1>(Size<1>(size0)),
    Texture<Type, 1>(x, Size<1>(ofs0), Size<1>(size0))
  {
  }
};

}  // namespace OpenGL
}  // namespace Cuda


#endif
