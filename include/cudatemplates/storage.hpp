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

#ifndef CUDA_STORAGE_H
#define CUDA_STORAGE_H


#include <cudatemplates/layout.hpp>


#define CUDA_SPECIALIZE_DIM_COMMON(Name, Dim)				\
  inline Name ## Dim ## D()						\
  {									\
  }									\
  inline Name ## Dim ## D(const Size<Dim> &_size):			\
    Layout<Type, Dim>(_size),						\
    CUDA_INIT_POINTER(Pointer<Type, Dim>(_size))			\
    Name<Type, Dim>(_size)						\
  {									\
  }									\
  inline Name ## Dim ## D(const Layout<Type, Dim> &layout):		\
    Layout<Type, Dim>(layout),						\
    CUDA_INIT_POINTER(Pointer<Type, Dim>(layout))			\
    Name<Type, Dim>(layout)						\
  {									\
  }									\
  inline void alloc()							\
  {									\
    Name<Type, Dim>::alloc();						\
  }

#define CUDA_SPECIALIZE_DIM(Name)					\
  template <class Type>							\
  class Name ## 1D: public Name<Type, 1>				\
  {									\
  public:								\
    CUDA_SPECIALIZE_DIM_COMMON(Name, 1)					\
    inline Name ## 1D(size_t s0):					\
      Layout<Type, 1>(Size<1>(s0)),					\
      CUDA_INIT_POINTER(Pointer<Type, 1>(Size<1>(s0)))			\
      Name<Type, 1>(Size<1>(s0))					\
    {									\
    }									\
    inline void alloc(size_t s0)					\
    {									\
      Storage<Type, 1>::alloc(Size<1>(s0));				\
    }									\
  };									\
  template <class Type>							\
  class Name ## 2D: public Name<Type, 2>				\
  {									\
  public:								\
    CUDA_SPECIALIZE_DIM_COMMON(Name, 2)					\
    inline Name ## 2D(size_t s0, size_t s1):				\
      Layout<Type, 2>(Size<2>(s0, s1)),					\
      CUDA_INIT_POINTER(Pointer<Type, 2>(Size<2>(s0, s1)))		\
      Name<Type, 2>(Size<2>(s0, s1))					\
    {									\
    }									\
    inline void alloc(size_t s0, size_t s1)				\
    {									\
      Storage<Type, 2>::alloc(Size<2>(s0, s1));				\
    }									\
  };									\
  template <class Type>							\
  class Name ## 3D: public Name<Type, 3>				\
  {									\
  public:								\
    CUDA_SPECIALIZE_DIM_COMMON(Name, 3)					\
    inline Name ## 3D(size_t s0, size_t s1, size_t s2):			\
      Layout<Type, 3>(Size<3>(s0, s1, s2)),				\
      CUDA_INIT_POINTER(Pointer<Type, 3>(Size<3>(s0, s1, s2)))		\
      Name<Type, 3>(Size<3>(s0, s1, s2))				\
    {									\
    }									\
    inline void alloc(size_t s0, size_t s1, size_t s2)			\
    {									\
      Storage<Type, 3>::alloc(Size<3>(s0, s1, s2));			\
    }									\
  };


#define CUDA_COPY_CONSTRUCTOR(Name1, Base)		\
  inline Name1(const Name1<Type, Dim> &x):		\
    Layout<Type, Dim>(x),				\
    CUDA_INIT_POINTER(Pointer<Type, Dim>(x))		\
    Base<Type, Dim>(x)					\
  {							\
    this->init();					\
    this->alloc();					\
    copy(*this, x);					\
  }							\
  template<class Name2>					\
  inline Name1(const Name2 &x):				\
    Layout<Type, Dim>(x),				\
    CUDA_INIT_POINTER(Pointer<Type, Dim>(x))		\
    Base<Type, Dim>(x)					\
  {							\
    this->init();					\
    this->alloc();					\
    copy(*this, x);					\
  }							\
  template<class Name2>					\
  inline Name1(const Name2 &x,				\
	       const Size<Dim> &ofs,			\
	       const Size<Dim> &size):			\
    Layout<Type, Dim>(size),				\
    CUDA_INIT_POINTER(Pointer<Type, Dim>(size))		\
    Base<Type, Dim>(size)				\
  {							\
    this->init();					\
    this->alloc();					\
    copy(*this, x, Size<Dim>(), ofs, size);		\
  }


namespace Cuda {

/**
   Class to represent memory that can be allocated and freed.
*/
template <class Type, unsigned Dim>
class Storage: virtual public Layout<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Storage()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size
  */
  inline Storage(const Size<Dim> &_size):
    Layout<Type, Dim>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout
  */
  inline Storage(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout)
  {
  }

  /**
     Destructor.
  */
  virtual ~Storage()
  {
  }

  /**
     Allocate memory.
  */
  virtual void alloc() = 0;

  /**
     Allocate memory.
     @param _size requested size
  */
  void alloc(const Size<Dim> &_size);

  /**
     Allocate memory.
     @param layout requested layout
  */
  void alloc(const Layout<Type, Dim> &layout);

  /**
     Free memory.
  */
  virtual void free() = 0;

  /**
     Get current layout.
     @return layout
  */
  const Layout<Type, Dim> &getLayout() const { return *this; }

  /**
     Initialize data structure.
  */
  virtual void init() = 0;

protected:
  /**
     Copy constructor.
  */
  Storage(const Storage &s): Layout<Type, Dim>(s) {}
};

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
alloc(const Size<Dim> &_size)
{
  free();
  this->setSize(_size);
  alloc();
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
alloc(const Layout<Type, Dim> &layout)
{
  free();
  this->setLayout(layout);
  alloc();
}

/**
   Class to represent memory that is accessible via a typed pointer.
   Host and device memory are accessible by pointers, but CUDA arrays are not.
*/
template <class Type, unsigned Dim>
class Pointer: virtual public Layout<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline Pointer():
    buffer(0)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size
  */
  inline Pointer(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    buffer(0)
  {
  }

  /**
     Constructor.
     @param layout requested layout
  */
  inline Pointer(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    buffer(0)
  {
  }

  /**
     Destructor.
  */
  virtual ~Pointer()
  {
  }

  /**
     Array index operator.
     @param i index
     @return value at index i
  */
  inline Type &operator[](size_t i) { return buffer[i]; }

  /**
     Array index operator.
     @param i index
     @return value at index i
  */
  inline Type &operator[](const Size<Dim> &i) { return buffer[this->getOffset(i)]; }

  /**
     Array index operator.
     @param i index
     @return value at index i (constant)
  */
  inline const Type &operator[](size_t i) const { return buffer[i]; }

  /**
     Array index operator.
     @param i index
     @return value at index i
  */
  inline const Type &operator[](const Size<Dim> &i) const { return buffer[this->getOffset(i)]; }

  /**
     Get buffer pointer.
     @return buffer pointer (constant)
  */
  inline const Type *getBuffer() const { return buffer; }

  /**
     Get buffer pointer.
     @return buffer pointer
  */
  inline Type *getBuffer() { return buffer; }

protected:
  Type *buffer;

  inline Pointer(const Pointer<Type, Dim> &x):
    Layout<Type, Dim>(x),
    buffer(0)  // subclasses must perform a copy operation
  {
  }
};

/**
   Class to represent memory that can be allocated and freed
   and is accessible via a typed pointer.
*/
template <class Type, unsigned Dim>
class PointerStorage: public Storage<Type, Dim>, virtual public Pointer<Type, Dim>
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
  */
  inline PointerStorage()
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size
  */
  inline PointerStorage(const Size<Dim> &_size):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    Storage<Type, Dim>(_size)
  {
  }

  /**
     Constructor.
     @param layout requested layout
  */
  inline PointerStorage(const Layout<Type, Dim> &layout):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    Storage<Type, Dim>(layout)
  {
  }

  /**
     Initialize data structure.
  */
  inline void init() { this->buffer = 0; }

protected:
  inline PointerStorage(const PointerStorage<Type, Dim> &x):
    Layout<Type, Dim>(x),
    Pointer<Type, Dim>(x),
    Storage<Type, Dim>(x)
  {
  }

};

}


#endif
