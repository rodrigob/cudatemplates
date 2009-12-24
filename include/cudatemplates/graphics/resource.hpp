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

}  // namespace Graphics
}  // namespace Cuda


#endif
