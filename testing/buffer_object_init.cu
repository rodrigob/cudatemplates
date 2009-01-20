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

#define GL_GLEXT_PROTOTYPES

#include <assert.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/hostmemoryreference.hpp>
#include <cudatemplates/opengl/bufferobject.hpp>
#include <cudatemplates/opengl/copy.hpp>


void
init_geometry(Cuda::OpenGL::BufferObject2D<float4> &bufobj_coords,
	      Cuda::OpenGL::BufferObject2D<float2> &bufobj_texcoords)
{
  assert(bufobj_coords.size == bufobj_texcoords.size);

  // create coordinate array:
  float4 *coords    = new float4[bufobj_coords   .getSize()];
  float2 *texcoords = new float2[bufobj_texcoords.getSize()];
  float4 *pc = coords;
  float2 *pt = texcoords;

  for(int i = 0; i < bufobj_coords.size[1]; ++i) {
    float fi = (float)i / bufobj_coords.size[1];

    for(int j = 0; j < bufobj_coords.size[0]; ++j) {
      float fj = (float)j / bufobj_coords.size[0];

      // coordinates:
      float dx = fi - 0.5;
      float dy = 0.5 - fj;
      float d = sqrt(dx * dx + dy * dy);
      float f = (d > 0) ? 2 * pow(d, 0.2) : 0;
      *(pc++) = make_float4(f * dx, f * dy, 0, 1);

      // texture coordinates:
      *(pt++) = make_float2(fi, fj);
    }
  }

  // create CUDA templates references to the arrays:
  Cuda::HostMemoryReference2D<float4> ref_coords   (bufobj_coords.size   , coords);
  Cuda::HostMemoryReference2D<float2> ref_texcoords(bufobj_texcoords.size, texcoords);

  // copy data to buffer objects:
  copy(bufobj_coords, ref_coords);
  copy(bufobj_texcoords, ref_texcoords);
  bufobj_coords.disconnect();
  bufobj_texcoords.disconnect();

  bufobj_coords.bind();
  glVertexPointer(4, GL_FLOAT, 0, 0);
  bufobj_texcoords.bind();
  glTexCoordPointer(2, GL_FLOAT, 0, 0);

  delete[] coords;
  delete[] texcoords;
}

void
init_topology(Cuda::OpenGL::BufferObject2D<int4> &bufobj_coordindex)
{
  // create coordinate index array:
  int4 *coordindex = new int4[bufobj_coordindex.getSize()];
  int4 *pi = coordindex;

  for(int i = 0; i < bufobj_coordindex.size[1]; ++i)
    for(int j = 0; j < bufobj_coordindex.size[0]; ++j) {
      int v0 = i * (bufobj_coordindex.size[0] + 1) + j;
      *(pi++) = make_int4(v0,
			  v0 + 1,
			  v0 + bufobj_coordindex.size[0] + 2,
			  v0 + bufobj_coordindex.size[0] + 1);
    }

  // create CUDA templates reference to the array:
  Cuda::HostMemoryReference2D<int4> ref_coordindex(bufobj_coordindex.size, coordindex);

  // copy data to buffer object:
  copy(bufobj_coordindex, ref_coordindex);

  bufobj_coordindex.disconnect();
  bufobj_coordindex.bind();

  delete[] coordindex;
}
