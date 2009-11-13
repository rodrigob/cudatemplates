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

#ifndef CUDA_COPY_GILREFERENCE_H
#define CUDA_COPY_GILREFERENCE_H


inline GilReference(const GilReference<Type, Dim> &x):
  Layout<Type, Dim>(x),
  Pointer<Type, Dim>(x),
  HostMemoryReference<Type, Dim>(x)
{
  this->init();
  this->realloc();
  copy(*this, x);
}

template<class Name>
inline GilReference(const Name &x):
  Layout<Type, Dim>(x),
  Pointer<Type, Dim>(x),
  HostMemoryReference<Type, Dim>(x)
{
  this->init();
  this->realloc();
  copy(*this, x);
}

template<class Name>
inline GilReference(const Name &x, const Size<Dim> &ofs, const Size<Dim> &size):
  Layout<Type, Dim>(size),
  Pointer<Type, Dim>(size),
  HostMemoryReference<Type, Dim>(size)
{
  this->init();
  this->realloc();
  copy(*this, x, Size<Dim>(), ofs, size);
}


#endif
