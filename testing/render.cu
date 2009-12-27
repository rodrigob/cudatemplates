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


#define USE_CUDA30 0


#include <iostream>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>

#include <cudatemplates/copy_constant.hpp>


typedef struct uchar3 PixelType;

#if USE_CUDA30

#include <cudatemplates/graphics/opengl/buffer.hpp>
typedef Cuda::Graphics::OpenGL::Buffer<PixelType, 2> FramebufferType;

#else

#include <cudatemplates/opengl/bufferobject.hpp>
typedef Cuda::OpenGL::BufferObject<PixelType, 2> FramebufferType;

#endif


using namespace std;


Cuda::Size<2> size0(512, 512);

FramebufferType *bufobj = 0;


void
reshape(int w, int h)
{
  glViewport(0, 0, w, h);
  bufobj->realloc(Cuda::Size<2>(w, h));
}

void
display()
{
  // clear framebuffer:
  Cuda::copy(*bufobj, make_uchar3(255, 255, 128));

  // save OpenGL state and reset transformation:
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // transfer pixels:
  glRasterPos2i(-1, -1);

#if USE_CUDA30
  bufobj->setState(Cuda::Graphics::Resource::STATE_GRAPHICS_BOUND);
#else
  bufobj->disconnect();
  bufobj->bind();
#endif

  glDrawPixels(bufobj->size[0], bufobj->size[1], GL_RGB, GL_UNSIGNED_BYTE, 0);

#if USE_CUDA30
  bufobj->setState(Cuda::Graphics::Resource::STATE_CUDA_MAPPED);
#else
  bufobj->unbind();
  bufobj->connect();
#endif

  // restore OpenGL state:
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glPopAttrib();

  // postprocess:
  glutSwapBuffers();
  glutPostRedisplay();
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
#if USE_CUDA30
    CUDA_CHECK(cudaGLSetGLDevice(0));
#endif

    // init GLUT:
    glutInit(&argc, argv);
    glutInitWindowSize(size0[0], size0[1]);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("CUDA render demo");
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    // create OpenGL buffer object:
    bufobj = new FramebufferType(size0, GL_PIXEL_UNPACK_BUFFER, GL_DYNAMIC_COPY);

    // enter main loop:
    glutMainLoop();
  }
  catch(const std::exception &e) {
    cerr << e.what() << endl;
    return 1;
  }

  delete bufobj;
  return 0;
}
