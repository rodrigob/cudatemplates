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

#include <assert.h>
#include <GL/glew.h>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/opengl/bufferobject.hpp>
#include <cudatemplates/opengl/copy.hpp>

#include "buffer_object.hpp"


const int BLOCK_SIZE = 8;


/**
   Integer division (rounding up the result).
*/
static inline int
div_up(int x, int y)
{
  return (x + y - 1) / y;
}

/**
   Compute vertex and texture coordinates.
*/
__global__ void
init_geometry_kernel(BufferObjectFloat4Type::KernelData coords, BufferObjectFloat2Type::KernelData texcoords)
{
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;

  if((i >= coords.size[1]) || (j >= coords.size[0]))
    return;

  int ofs = i * coords.size[0] + j;

  float fi = (float)i / (coords.size[1] - 1);
  float fj = (float)j / (coords.size[0] - 1);

  // coordinates:
  float dx = fj - 0.5f;
  float dy = 0.5f - fi;
  float d = sqrtf(dx * dx + dy * dy);
  float f = (d > 0) ? 2 * powf(d, 0.2f) : 0;
  coords.data[ofs] = make_float4(f * dx, f * dy, 0, 1);

  // texture coordinates:
  texcoords.data[ofs] = make_float2(fj, fi);
}

/**
   Init mesh geometry on GPU.
*/
void
init_geometry(BufferObjectFloat4Type &bufobj_coords, BufferObjectFloat2Type &bufobj_texcoords)
{
  assert(bufobj_coords.size == bufobj_texcoords.size);

  // launch kernel:
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 dimGrid(div_up(bufobj_coords.size[0], dimBlock.x), div_up(bufobj_coords.size[1], dimBlock.y), 1);
  init_geometry_kernel<<<dimBlock, dimGrid>>>(bufobj_coords, bufobj_texcoords);
  cudaThreadSynchronize();

  // use coordinates for OpenGL:
  bufobj_coords.disconnect();
  bufobj_coords.bind();
  glVertexPointer(4, GL_FLOAT, 0, 0);

  // use texture coordinates for OpenGL:
  bufobj_texcoords.disconnect();
  bufobj_texcoords.bind();
  glTexCoordPointer(2, GL_FLOAT, 0, 0);
}

/**
   Init mesh topology on CPU.
*/
void
init_topology(BufferObjectInt4Type &bufobj_coordindex)
{
  // create topology array:
  Cuda::HostMemoryHeap2D<int4> coordindex(bufobj_coordindex.size);
  int4 *pi = coordindex.getBuffer();

  for(unsigned i = 0; i < bufobj_coordindex.size[1]; ++i)
    for(unsigned j = 0; j < bufobj_coordindex.size[0]; ++j) {
      int v0 = i * (bufobj_coordindex.size[0] + 1) + j;
      *(pi++) = make_int4(v0,
			  v0 + 1,
			  v0 + bufobj_coordindex.size[0] + 2,
			  v0 + bufobj_coordindex.size[0] + 1);
    }

  // copy data to buffer object:
  copy(bufobj_coordindex, coordindex);

  bufobj_coordindex.disconnect();
  bufobj_coordindex.bind();
}
