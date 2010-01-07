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


#define USE_CUDA30  0
#define USE_TEXTURE 0


#if USE_TEXTURE && !USE_CUDA30
#error binding OpenGL textures is only supported in CUDA3.0
#endif


#include <time.h>

#include <iostream>
#include <vector>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/copy_constant.hpp>
#include <cudatemplates/devicememorylinear.hpp>


typedef struct uchar4 PixelType;
const PixelType background(make_uchar4(255, 255, 128, 0));


#if USE_CUDA30

#include <cudatemplates/graphics/opengl/buffer.hpp>
#include <cudatemplates/graphics/opengl/texture.hpp>
typedef Cuda::Graphics::OpenGL::Buffer<PixelType, 2> BufferType;
typedef Cuda::Graphics::OpenGL::Texture<PixelType, 2> TextureType;

#else

#include <cudatemplates/opengl/bufferobject.hpp>
#include <cudatemplates/opengl/texture.hpp>
typedef Cuda::OpenGL::BufferObject<PixelType, 2> BufferType;
typedef Cuda::OpenGL::Texture<PixelType, 2> TextureType;

#endif

typedef Cuda::DeviceMemoryLinear<PixelType, 2> MemoryType;


using namespace std;


Cuda::Size<2> size0(1024, 768);


#if USE_TEXTURE

TextureType *texobj;
MemoryType *mem;

#else

BufferType *bufobj;

#endif


__global__ void
render_kernel(Cuda::DeviceMemory<PixelType, 2>::KernelData dst)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int ofs = x + y * dst.size[0];
  dst.data[ofs] = make_uchar4(255, 255, 128, 0);
}

void
render()
{
  dim3 blockDim(16, 16);
  dim3 gridDim(size0[0] / blockDim.x, size0[1] / blockDim.y);

#if USE_TEXTURE
  // Cuda::DeviceMemory<PixelType, 2> &dst(*mem);
  render_kernel<<<gridDim, blockDim>>>(*mem);
  texobj->setState(Cuda::Graphics::Resource::STATE_CUDA_MAPPED);
  Cuda::copy(*texobj, *mem);
#else
  bufobj->unbind();
  bufobj->connect();
  // Cuda::DeviceMemory<PixelType, 2> &dst(*bufobj);
  render_kernel<<<gridDim, blockDim>>>(*bufobj);
#endif

  /*
#if USE_CUDA30
  dst.setState(Cuda::Graphics::Resource::STATE_CUDA_MAPPED);
#else
  dst.unbind();
  dst.connect();
#endif
  */

}

/**
  Clear framebuffer.
*/
/*
void
clear()
{
#if USE_TEXTURE

  // Cuda::copy(*mem, background);
  texobj->setState(Cuda::Graphics::Resource::STATE_CUDA_MAPPED);
  Cuda::copy(*texobj, *mem);

#else

#if USE_CUDA30
  bufobj->setState(Cuda::Graphics::Resource::STATE_CUDA_MAPPED);
#else
  bufobj->unbind();
  bufobj->connect();
#endif

  Cuda::copy(*bufobj, background);

#endif
}
*/

void
display()
{
  // glClear(GL_COLOR_BUFFER_BIT);

  render();
  
  // transfer pixels:

#if USE_TEXTURE

  texobj->setState(Cuda::Graphics::Resource::STATE_GRAPHICS_BOUND);
  glBegin(GL_POLYGON);
  glTexCoord2f(0, 0); glVertex2f(-1, -1);
  glTexCoord2f(1, 0); glVertex2f( 1, -1);
  glTexCoord2f(1, 1); glVertex2f( 1,  1);
  glTexCoord2f(0, 1); glVertex2f(-1,  1);
  glEnd();

#else

  glRasterPos2i(-1, -1);

#if USE_CUDA30
  bufobj->setState(Cuda::Graphics::Resource::STATE_GRAPHICS_BOUND);
#else
  bufobj->disconnect();
  bufobj->bind();
#endif

  glDrawPixels(bufobj->size[0], bufobj->size[1], GL_RGBA, GL_UNSIGNED_BYTE, 0);

#endif

  // postprocess:
  glutSwapBuffers();
  glutPostRedisplay();

  // timing:
  static int count = 0;
  static time_t tprev = 0;

  ++count;
  time_t t = time(0);

  if(tprev == 0)
    tprev = t;

  if(t == tprev)
    return;

  float gbps = (long long)(size0[0] * size0[1] * sizeof(PixelType) * count) / (float)(1 << 30);
  cout << count << " frames/sec = " << gbps << " GB/sec\n";
  count = 0;
  tprev = t;
}

void
keyboard(unsigned char c, int, int)
{
  if(c == 0x1b)
    exit(0);
}

int
main(int argc, char *argv[])
{
  try {
    // init GLUT:
    glutInit(&argc, argv);
    glutInitWindowSize(size0[0], size0[1]);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutCreateWindow("CUDA render demo");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

#if USE_CUDA30
    CUDA_CHECK(cudaGLSetGLDevice(0));
#endif

#if USE_TEXTURE

    texobj = new TextureType(size0, cudaGraphicsMapFlagsWriteDiscard);
    mem = new MemoryType(size0);

    // enable textures and set parameters:
    texobj->setState(Cuda::Graphics::Resource::STATE_GRAPHICS_BOUND);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

#else  // USE_TEXTURE

#if USE_CUDA30
    bufobj = new BufferType(size0, GL_PIXEL_UNPACK_BUFFER, GL_STREAM_DRAW, cudaGraphicsMapFlagsWriteDiscard);
#else
    bufobj = new BufferType(size0, GL_PIXEL_UNPACK_BUFFER, GL_STREAM_DRAW);
#endif

#endif  // USE_TEXTURE

    // clear();
    render();

    // reset OpenGL transformation:
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // enter main loop:
    glutMainLoop();
  }
  catch(const std::exception &e) {
    cerr << e.what() << endl;
    return 1;
  }

  return 0;
}
