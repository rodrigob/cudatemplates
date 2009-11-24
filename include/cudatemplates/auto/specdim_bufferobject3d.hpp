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

#ifndef CUDA_BUFFEROBJECT3D_H
#define CUDA_BUFFEROBJECT3D_H


#include <cudatemplates/opengl/bufferobject.hpp>


namespace Cuda {
namespace OpenGL {

/**
   BufferObject template specialized for 3 dimension(s).
*/
template <class Type>
class BufferObject3D:
    virtual public Layout<Type, 3>,
    virtual public Pointer<Type, 3>,
    public BufferObject<Type, 3>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline BufferObject3D()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
  */
  inline BufferObject3D(const Size<3> &_size, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW):
    Layout<Type, 3>(_size),
    Pointer<Type, 3>(_size),
    BufferObject<Type, 3>(_size, t, u)
  {
  }

  /**
     Constructor.
     @param layout requested layout of memory block.
  */
  inline BufferObject3D(const Layout<Type, 3> &layout, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW):
    Layout<Type, 3>(layout),
    Pointer<Type, 3>(layout),
    BufferObject<Type, 3>(layout, t, u)
  {
  }

  /**
     Constructor.
  */
  inline BufferObject3D(size_t size0, size_t size1, size_t size2, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW):
    Layout<Type, 3>(Size<3>(size0, size1, size2)),
    Pointer<Type, 3>(Size<3>(size0, size1, size2)),
    BufferObject<Type, 3>(Size<3>(size0, size1, size2), t, u)
  {
  }

  /**
     Copy constructor.
     @param x instance of BufferObject3D to be copied
  */
  inline BufferObject3D(const BufferObject3D<Type> &x):
    Layout<Type, 3>(x),
    Pointer<Type, 3>(x),
    BufferObject<Type, 3>(x)
  {
  }

  /**
     Constructor.
     Initialization from different type.
     @param x instance of different type to be copied
  */
  template<class Name>
    inline BufferObject3D(const Name &x):
    Layout<Type, 3>(x),
    Pointer<Type, 3>(x),
    BufferObject<Type, 3>(x)
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
    inline BufferObject3D(const Name &x, const Size<3> &ofs, const Size<3> &size):
    Layout<Type, 3>(size),
    Pointer<Type, 3>(size),
    BufferObject<Type, 3>(x, ofs, size)
  {
  }

  /**
     Constructor.
     Initialization of region from same or different type.
     @param x instance to be copied
     @param ofs0, ofs1, ofs2 offset of region
     @param size0, size1, size2 size of region
  */
  template<class Name>
    inline BufferObject3D(const Name &x, size_t ofs0, size_t ofs1, size_t ofs2, size_t size0, size_t size1, size_t size2):
    Layout<Type, 3>(Size<3>(size0, size1, size2)),
    Pointer<Type, 3>(Size<3>(size0, size1, size2)),
    BufferObject<Type, 3>(x, Size<3>(ofs0, ofs1, ofs2), Size<3>(size0, size1, size2))
  {
  }

  /**
     Allocate memory.
  */
  inline void alloc()
  {
    BufferObject<Type, 3>::alloc();
  }

  /**
     Allocate memory.
     @param _size size to be allocated
  */
  inline void alloc(const Size<3> &_size)
  {
    Storage<Type, 3>::alloc(_size);
  }

  /**
     Allocate memory.
     size0, size1, size2 size to be allocated
  */
  inline void alloc(size_t size0, size_t size1, size_t size2)
  {
    Storage<Type, 3>::alloc(Size<3>(size0, size1, size2));
  }

  /**
     Re-allocate memory.
  */
  inline void realloc()
  {
    BufferObject<Type, 3>::realloc();
  }

  /**
     Re-allocate memory.
     @param _size size to be allocated
  */
  inline void realloc(const Size<3> &_size)
  {
    Storage<Type, 3>::realloc(_size);
  }

  /**
     Re-allocate memory.
     size0, size1, size2 size to be allocated
  */
  inline void realloc(size_t size0, size_t size1, size_t size2)
  {
    Storage<Type, 3>::realloc(Size<3>(size0, size1, size2));
  }

};

}  // namespace OpenGL
}  // namespace Cuda


#endif
