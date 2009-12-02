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


namespace Cuda {

/**
   Class to represent memory that can be allocated and freed.
   This is used as a virtual base class for all types of memory for which the
   CUDA templates should perform their own memory management (i.e.,
   allocation and deallocation).
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
     See Storage::realloc(const Layout<Type, Dim> &) for possible performance implications.
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
     This method also initializes spacing and region.
  */
  void alloc();

  /**
     Allocate memory.
     This method also initializes spacing and region.
     @param _size requested size
  */
  void alloc(const Size<Dim> &_size);

  /**
     Allocate memory.
     This method also initializes spacing and region.
     @param layout requested layout
  */
  void alloc(const Layout<Type, Dim> &layout);

  /**
     Reallocate memory.
     This method doesn't modify spacing and region.
  */
  virtual void realloc() = 0;

  /**
     Reallocate memory.
     This method doesn't modify spacing and region.
     @param _size requested size
  */
  void realloc(const Size<Dim> &_size);

  /**
     Reallocate memory.
     This uses the requested layout regardless of constraints imposed by
     subclasses (such as DeviceMemoryPitched). Use with care to avoid
     performance penalties!
     This method doesn't modify spacing and region.
     @param layout requested layout
  */
  void realloc(const Layout<Type, Dim> &layout);

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
     Subclasses should use this method to initialize their internal data
     representation to a defined "empty" state.
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
alloc()
{
  realloc();
  this->initSpacingRegion();
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
alloc(const Size<Dim> &_size)
{
  realloc(_size);
  this->initSpacingRegion();
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
alloc(const Layout<Type, Dim> &layout)
{
  realloc(layout);
  this->initSpacingRegion();
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
realloc(const Size<Dim> &_size)
{
  if(_size == this->size)
    return;

  free();
  this->setSize(_size);
  realloc();
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
realloc(const Layout<Type, Dim> &layout)
{
  if(layout == *this)
    return;

  free();
  this->setLayout(layout);
  realloc();
}

}  // namespace Cuda


#endif
