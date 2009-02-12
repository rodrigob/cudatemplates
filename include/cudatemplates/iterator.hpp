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
class Iterator: public SizeBase<Dim>
{
  CUDA_STATIC_ASSERT(Dim > 0);

public:
  /**
     Constructor iterator for given size.
     @param _imax maximum
  */
  inline Iterator(const SizeBase<Dim> &_imax):
    imax(_imax)
  {
  }

  /**
     Constructor iterator for given range.
     @param _imin minimum
     @param _imax maximum
  */
  inline Iterator(const SizeBase<Dim> &_imin, const SizeBase<Dim> &_imax):
    SizeBase<Dim>(_imin),
    imin(_imin), imax(_imax)
  {
  }

  inline const Iterator &operator++()
  {
    for(unsigned j = 0; j < Dim; ++j) {
      if(++(*this)[j] < imax[j])
	return *this;
      
      (*this)[j] = imin[j];
    }
    
    setEnd();
    return *this;
  }

  /**
     Set iterator to begin of data.
     @return updated iterator
  */
  inline const Iterator &setBegin()
  {
    SizeBase<Dim>::operator=(imin);
    return *this;
  }

  /**
     Set iterator to end of data.
     @return updated iterator
  */
  inline const Iterator &setEnd()
  {
    SizeBase<Dim>::operator=(imax);
    return *this;
  }

protected:
  /**
     The iterator range.
     imin is inclusive, imax is exclusive.
     Setting the index to imax indicates the "end" condition.
  */
  SizeBase<Dim> imin, imax;
};

}  // namespace Cuda


#endif
