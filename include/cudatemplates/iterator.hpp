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

#ifndef CUDA_ITERATOR_H
#define CUDA_ITERATOR_H


#include <cudatemplates/size.hpp>


namespace Cuda {

/**
   Generic iterator over n-dimensional data.
*/
template <unsigned Dim>
class Iterator: public Size<Dim>
{
  CUDA_STATIC_ASSERT(Dim > 0);

public:
  /**
     Constructor iterator for given size.
     @param _imax maximum
  */
  inline Iterator(const Size<Dim> &_imax):
    imax(_imax)
  {
  }

  /**
     Constructor iterator for given range.
     @param _imin minimum
     @param _imax maximum
  */
  inline Iterator(const Size<Dim> &_imin, const Size<Dim> &_imax):
    Size<Dim>(_imin),
    imin(_imin), imax(_imax)
  {
  }

  inline const Iterator &operator++()
  {
    for(unsigned j = 0; j < Dim; ++j) {
      if(++this->size[j] < imax[j])
	break;
      
      this->size[j] = imin[j];
    }
    
    return *this;
  }

  /**
     Set iterator to begin of data.
     @return updated iterator
  */
  inline const Iterator &setBegin() { *this = imin; return *this; }

  /**
     Set iterator to end of data.
     @return updated iterator
  */
  inline const Iterator &setEnd() { *this = imax; return *this; }

protected:
  /**
     The iterator range.
  */
  Size<Dim> imin, imax;
};

}  // namespace Cuda


#endif
