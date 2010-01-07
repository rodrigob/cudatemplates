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

#ifndef BUFFER_OBJECT_H
#define BUFFER_OBJECT_H


#define USE_CUDA30 0


#if USE_CUDA30

#define CUDA_GRAPHICS_COMPATIBILITY

#include <cudatemplates/graphics/copy.hpp>
#include <cudatemplates/graphics/opengl/buffer.hpp>

#else

#include <cudatemplates/opengl/bufferobject.hpp>
#include <cudatemplates/opengl/copy.hpp>

#endif


typedef struct uchar3 PixelType;

#if USE_CUDA30

typedef Cuda::Graphics::OpenGL::Buffer<PixelType, 2> BufferObjectPixelType;
typedef Cuda::Graphics::OpenGL::Buffer<int4, 2> BufferObjectInt4Type;
typedef Cuda::Graphics::OpenGL::Buffer<float2, 2> BufferObjectFloat2Type;
typedef Cuda::Graphics::OpenGL::Buffer<float4, 2> BufferObjectFloat4Type;

#else

typedef Cuda::OpenGL::BufferObject<PixelType, 2> BufferObjectPixelType;
typedef Cuda::OpenGL::BufferObject<int4, 2> BufferObjectInt4Type;
typedef Cuda::OpenGL::BufferObject<float2, 2> BufferObjectFloat2Type;
typedef Cuda::OpenGL::BufferObject<float4, 2> BufferObjectFloat4Type;

#endif


#endif
