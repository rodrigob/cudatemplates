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
  // void alloc();

  /**
     Allocate memory.
     This method also initializes spacing and region.
     The object is required to be empty (use "free" or "realloc" if this is not
     the case).
     @param _size requested size
  */
  void alloc(const Size<Dim> &_size);

  /**
     Allocate memory.
     This method also initializes spacing and region.
     The object is required to be empty (use "free" or "realloc" if this is not
     the case).
     @param layout requested layout
  */
  void alloc(const Layout<Type, Dim> &layout);

  /**
     Free memory.
  */
  void free();

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

  /**
     Determine if memory is currently allocated.
  */
  inline bool isAllocated() const { return !empty(); }

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

protected:
  /**
     Copy constructor.
  */
  Storage(const Storage &s): Layout<Type, Dim>(s) {}

private:
  void allocCheckSize() const;

  /**
     Allocate memory.
     Subclasses must overload this method to do the actual work for memory
     allocation. An implementation of this method can assume the following:
     *) no memory is currently allocated
     *) the members of the Layout base class are set according to the requested
     allocation
     *) the requested size differs from the previously allocated size (if any)
  */
  virtual void allocInternal() = 0;

  /**
     Free memory.
     Subclasses must overload this method to do the actual work for memory
     deallocation. An implementation of this method can assume the following:
     *) memory is currently allocated
  */
  virtual void freeInternal() = 0;

  static void reallocCheckSize(const Size<Dim> &_size);
};

  /*
template <class Type, unsigned Dim>
void Storage<Type, Dim>::
alloc()
{
  if(size.empty())
    CUDA_ERROR("trying to allocate empty object");

  if(!empty())
    freeInternal();

  allocInternal();

  realloc();
  this->initSpacingRegion();
}
  */

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
allocCheckSize() const
{
  if(!size.empty())
    CUDA_ERROR("\"alloc\" requires empty object (use \"free\" or \"realloc\" instead)");
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
alloc(const Size<Dim> &_size)
{
  allocCheckSize();
  realloc(_size);
  this->initSpacingRegion();
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
alloc(const Layout<Type, Dim> &layout)
{
  allocCheckSize();
  realloc(layout);
  this->initSpacingRegion();
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
free()
{
  if(!isAllocated())
    return;

  freeInternal();

  for(size_t i = Dim; i--;)
    size[i] = 0;
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
reallocCheckSize(const Size<Dim> &_size) const
{
  if(_size.empty())
    CUDA_ERROR("trying to allocate empty object");
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
realloc(const Size<Dim> &_size)
{
  reallocCheckSize(_size);

  if(_size == this->size)
    return;

  if(isAllocated())
    freeInternal();

  this->setSize(_size);
  allocInternal();
}

template <class Type, unsigned Dim>
void Storage<Type, Dim>::
realloc(const Layout<Type, Dim> &layout)
{
  reallocCheckSize(layout.size);

  if(layout == *this)
    return;

  if(isAllocated())
    freeInternal();

  this->setLayout(layout);
  allocInternal();
}

}  // namespace Cuda


#endif
