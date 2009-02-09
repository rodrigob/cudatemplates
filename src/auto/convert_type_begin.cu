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


/*

  TODO: implement the following optimizations:

  1.) In the (very likely) case that the strides for source and destination
  data are the same, the index only needs to be computed once.

  2.) Conversions between signed and unsigned variants of the same type can
  more efficiently be handled by a reinterpret_cast and a non-converting copy.
  
*/


#include <cudatemplates/convert.hpp>
#include <cudatemplates/devicememory.hpp>
