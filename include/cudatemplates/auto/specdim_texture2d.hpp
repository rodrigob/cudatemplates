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

#ifndef CUDA_TEXTURE2D_H
#define CUDA_TEXTURE2D_H


#include <cudatemplates/opengl/texture.hpp>


namespace Cuda {
namespace OpenGL {

/**
   Texture template specialized for 2 dimension(s).
*/
template <class Type>
class Texture2D:
    virtual public Layout<Type, 2>,
    virtual public Pointer<Type, 2>,
    public Texture<Type, 2>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Texture2D()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
  */
  inline Texture2D(const Size<2> &_size):
    Layout<Type, 2>(_size),
    Pointer<Type, 2>(_size),
    Texture<Type, 2>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout of memory block.
  */
  inline Texture2D(const Layout<Type, 2> &layout):
    Layout<Type, 2>(layout),
    Pointer<Type, 2>(layout),
    Texture<Type, 2>(layout)
  {
  }

  /**
     Constructor.
  */
  inline Texture2D(size_t size0, size_t size1):
    Layout<Type, 2>(Size<2>(size0, size1)),
    Pointer<Type, 2>(Size<2>(size0, size1)),
    Texture<Type, 2>(Size<2>(size0, size1))
  {
  }

  /**
     Copy constructor.
     @param x instance of Texture2D to be copied
  */
  inline Texture2D(const Texture2D<Type> &x):
    Layout<Type, 2>(x),
    Pointer<Type, 2>(x),
    Texture<Type, 2>(x)
  {
  }

  /**
     Constructor.
     Initialization from different type.
     @param x instance of different type to be copied
  */
  template<class Name>
    inline Texture2D(const Name &x):
    Layout<Type, 2>(x),
    Pointer<Type, 2>(x),
    Texture<Type, 2>(x)
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
    inline Texture2D(const Name &x, const Size<2> &ofs, const Size<2> &size):
    Layout<Type, 2>(size),
    Pointer<Type, 2>(size),
    Texture<Type, 2>(x, ofs, size)
  {
  }

  /**
     Constructor.
     Initialization of region from same or different type.
     @param x instance to be copied
     @param ofs0, ofs1 offset of region
     @param size0, size1 size of region
  */
  template<class Name>
    inline Texture2D(const Name &x, size_t ofs0, size_t ofs1, size_t size0, size_t size1):
    Layout<Type, 2>(Size<2>(size0, size1)),
    Pointer<Type, 2>(Size<2>(size0, size1)),
    Texture<Type, 2>(x, Size<2>(ofs0, ofs1), Size<2>(size0, size1))
  {
  }
};

}  // namespace OpenGL
}  // namespace Cuda


#endif
