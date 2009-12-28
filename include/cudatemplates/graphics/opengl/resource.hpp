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

#ifndef CUDA_GRAPHICS_OPENGL_RESOURCE_H
#define CUDA_GRAPHICS_OPENGL_RESOURCE_H


// #include <assert.h>

/*
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <cudatemplates/devicememory.hpp>
#include <cudatemplates/error.hpp>
#include <cudatemplates/layout.hpp>
#include <cudatemplates/opengl/error.hpp>
*/
#include <cudatemplates/graphics/resource.hpp>


namespace Cuda {
namespace Graphics {
namespace OpenGL {

class Resource: public Graphics::Resource
{
protected:
  /**
     OpenGL object name.
  */
  GLuint name;

  /**
     Constructor.
     @param f map flags
  */
  Resource(unsigned int f):
    Graphics::Resource(f),
    bound(false)
  {
  }

private:
  /**
     Flag to indicate bind status.
  */
  bool bound;

  /**
     Check bind status.
     @return true if image is currently bound for use with OpenGL
  */
  bool isBound() const
  {
    return bound;
  }

  /**
     Bind the image object to the target specified in the constructor.
  */
  void bindObject()
  {
    bindObjectInternal();
    bound = true;
  }

  virtual void bindObjectInternal() = 0;

  /**
     Unbind the image object from the target specified in the constructor.
  */
  inline void unbindObject()
  {
    bound = false;
    unbindObjectInternal();
  }

  virtual void unbindObjectInternal() = 0;
};

}  // namespace OpenGL
}  // namespace Graphics
}  // namespace Cuda


#endif
