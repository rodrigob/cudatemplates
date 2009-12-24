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

#ifndef CUDA_GRAPHICS_RESOURCE_H
#define CUDA_GRAPHICS_RESOURCE_H


#include <assert.h>

#include <cuda_runtime_api.h>

#include <cudatemplates/error.hpp>
#include <cudatemplates/layout.hpp>
#include <cudatemplates/opengl/error.hpp>


namespace Cuda {
namespace Graphics {

class Resource
{
public:
  typedef enum {
    STATE_GRAPHICS_BOUND = 1,
    STATE_UNUSED,
    STATE_CUDA_REGISTERED,
    STATE_CUDA_MAPPED
  } state_t;

  Resource():
    resource(0)
  {
  }

  virtual ~Resource()
  {
    setState(STATE_UNUSED);
  }

  inline void setMapFlags(unsigned int flags)
  {
    CUDA_CHECK(cudaGraphicsResourceSetMapFlags(resource, flags));
  }

  /**
     Get object state.
     @return current object state
  */
  state_t getState() const
  {
    if(isMapped())
      return STATE_CUDA_MAPPED;

    if(isRegistered())
      return STATE_CUDA_REGISTERED;

    if(isBound())
      return STATE_GRAPHICS_BOUND;

    return STATE_UNUSED;
  }

  /**
     Set object state.
     @param state_new new object state
     @return old object state
  */
  state_t setState(state_t state_new)
  {
    state_t state_old = getState();

    if(state_new > state_old) {
#define COND(state) if((state_old < STATE_ ## state) && (state_new >= STATE_ ## state))
      COND(UNUSED) unbindObject();
      COND(CUDA_REGISTERED) registerObject();
      COND(CUDA_MAPPED) mapObject();
#undef COND
    }
    else if(state_new < state_old) {
#define COND(state) if((state_old > STATE_ ## state) && (state_new <= STATE_ ## state))
      COND(CUDA_REGISTERED) unmapObject();
      COND(UNUSED) unregisterObject();
      COND(GRAPHICS_BOUND) bindObject();
#undef COND
    }

    return state_old;
  }

#ifdef CUDA_GRAPHICS_COMPATIBILITY
  inline void bind() { setState(STATE_GRAPHICS_BOUND); }
  inline void connect() { setState(STATE_CUDA_MAPPED); }
  inline void disconnect() { setState(STATE_UNUSED); }
#endif

protected:
  cudaGraphicsResource *resource;

private:
  /**
     Bind object for use with graphics API.
     This should only be called from setState() to avoid invalid state transitions.
  */
  virtual void bindObject() = 0;

  /**
     Check bound state.
     @return true if object is currently bound for use with graphics API.
  */
  virtual bool isBound() const = 0;

  /**
     Check mapped state.
     @return true if object is currently mapped for use with CUDA.
  */
  virtual bool isMapped() const = 0;

  /**
     Check registered state.
     @return true if object is currently registered for use with CUDA.
  */
  bool isRegistered() const
  {
    return resource != 0;
  }

  /**
     Object-specific part of map action.
  */
  virtual void mapInternal() = 0;

  /**
     Map object for use with CUDA.
     This should only be called from setState() to avoid invalid state transitions.
  */
  void mapObject()
  {
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource, 0));
    mapInternal();
  }

  /**
     Register object for use with CUDA.
     This should only be called from setState() to avoid invalid state transitions.
  */
  virtual void registerObject() = 0;

  /**
     Unbind object.
     This should only be called from setState() to avoid invalid state transitions.
  */
  virtual void unbindObject() = 0;

  /**
     Object-specific part of unmap action.
  */
  virtual void unmapInternal() = 0;

  /**
     Unmap object.
     This should only be called from setState() to avoid invalid state transitions.
  */
  void unmapObject()
  {
    unmapInternal();
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, 0));
  }

  /**
     Unregister object.
     This should only be called from setState() to avoid invalid state transitions.
  */
  void unregisterObject()
  {
    CUDA_CHECK(cudaGraphicsUnregisterResource(resource));
    resource = 0;
  }
};

namespace OpenGL {

template <class Type, unsigned Dim>
class Buffer: public DeviceMemoryStorage<Type, Dim>, public Resource
{
public:
#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
  /**
     Default constructor.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline Buffer(GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW, unsigned int f = 0):
    bufname(0), target(t), usage(u), flags(f), bound(false)
  {
  }
#endif

  /**
     Constructor.
     @param _size requested size of memory block.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline Buffer(const Size<Dim> &_size, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW, unsigned int f = 0):
    Layout<Type, Dim>(_size),
    Pointer<Type, Dim>(_size),
    DeviceMemoryStorage<Type, Dim>(_size),
    bufname(0), target(t), usage(u), flags(f), bound(false)
  {
    realloc();
  }

  /**
     Constructor.
     @param layout requested size of memory block.
     @param t target to which the buffer object is bound
     @param u usage pattern of the data store
  */
  inline Buffer(const Layout<Type, Dim> &layout, GLenum t = GL_ARRAY_BUFFER, GLenum u = GL_STATIC_DRAW, unsigned int f = 0):
    Layout<Type, Dim>(layout),
    Pointer<Type, Dim>(layout),
    DeviceMemoryStorage<Type, Dim>(layout),
    bufname(0), target(t), usage(u), flags(f), bound(false)
  {
    realloc();
  }

  ~Buffer();

  // #include "auto/copy_opengl_bufferobject.hpp"

  /**
     Bind the buffer object to the given target.
     @param t target to which the buffer object should be bound
  */
  /*
  inline void bind(GLenum t)
  {
    CUDA_OPENGL_CHECK(glBindBuffer(t, this->bufname));
  }
  */

  /**
     Free buffer memory.
  */
  void free();

  /**
     Allocate buffer memory.
  */
  void realloc();

  /**
     Allocate buffer memory.
     @_size size to be allocated
  */
  inline void realloc(const Size<Dim> &_size)
  {
    DeviceMemoryStorage<Type, Dim>::realloc(_size);
  }

  /**
     Set buffer object target.
     If the buffer object is currently bound to a different target, it will be
     unbound and then bound to the new target.
     @param target_new new target
     @return old target
  */
  GLenum setTarget(GLenum target_new)
  {
    if(target_new == target)
      return target;

    GLenum target_old = target;
    state_t state = getState();

    if(state == STATE_GRAPHICS_BOUND)
      setState(STATE_UNUSED);

    target = target_new;
    setState(state);
    return target_old;
  }

  /**
     Unbind the buffer object from the given target.
     @param t target from which the buffer object should be unbound
  */
  /*
  inline static void unbind(GLenum t) {
    CUDA_OPENGL_CHECK(glBindBuffer(t, 0));
  }
  */

#ifdef CUDA_GRAPHICS_COMPATIBILITY
  inline void bind() { Resource::bind(); }
#endif

private:
  /**
     Buffer object name.
  */
  GLuint bufname;

  /**
     Specifies the target to which the buffer object is bound.
  */
  GLenum target;

  /**
     Specifies the expected usage pattern of the data store.
  */
  GLenum usage;

  /**
     Flags for buffer registration in CUDA.
  */
  unsigned int flags;

  /**
     Flag to indicate bind status.
  */
  bool bound;

  /**
     Bind the buffer object to the target specified in the constructor.
  */
  void bindObject()
  {
    CUDA_OPENGL_CHECK(glBindBuffer(this->target, this->bufname));
    bound = true;
  }

  /**
     Check bind status.
     @return true if buffer is currently bound for use with OpenGL
  */
  bool isBound() const
  {
    return bound;
  }

  /**
     Check map status.
     @return true if buffer is currently mapped for use with CUDA
  */
  bool isMapped() const
  {
    return this->buffer != 0;
  }

  /**
     Object-specific part of map action.
  */
  void mapInternal()
  {
    size_t bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&this->buffer, &bytes, resource));
    
    if(this->buffer == 0)
      CUDA_ERROR("map buffer object failed");

    if(bytes != this->getBytes())
      CUDA_ERROR("size mismatch");
  }

  /**
     Register OpenGL buffer object for use with CUDA.
  */
  void registerObject()
  {
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&resource, bufname, flags));

    if(this->resource == 0)
      CUDA_ERROR("register buffer object failed");
  }

  /**
     Unbind the buffer object from the target specified in the constructor.
  */
  inline void unbindObject()
  {
    bound = false;
    CUDA_OPENGL_CHECK(glBindBuffer(this->target, 0));
  }

  /**
     Object-specific part of unmap action.
  */
  void unmapInternal()
  {
    this->buffer = 0;
  }
};

template <class Type, unsigned Dim>
void Buffer<Type, Dim>::
realloc()
{
  this->free();
  this->setPitch(0);
  size_t p = 1;

  for(size_t i = Dim; i--;)
    p *= this->size[i];

  CUDA_OPENGL_CHECK(glGenBuffers(1, &(this->bufname)));

  if(this->bufname == 0)
    CUDA_ERROR("generate buffer object failed");

  setState(STATE_GRAPHICS_BOUND);
  CUDA_OPENGL_CHECK(glBufferData(this->target, p * sizeof(Type), 0, this->usage));
  setState(STATE_CUDA_MAPPED);
}

template <class Type, unsigned Dim>
void Buffer<Type, Dim>::
free()
{
  if(this->bufname == 0)
    return;

  setState(STATE_UNUSED);
  glDeleteBuffers(1, &(this->bufname));
  bufname = 0;
}

template <class Type, unsigned Dim>
Buffer<Type, Dim>::
~Buffer()
{
  this->free();
}

}  // namespace OpenGL
}  // namespace Graphics
}  // namespace Cuda


#endif
