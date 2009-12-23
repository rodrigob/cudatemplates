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

  This example demonstrates three different use cases for OpenGL buffer objects
  in CUDA:

  *) using a buffer object as temporary storage for pixel data (bufobj_image)

  *) initializing a buffer object in a CUDA kernel and using the data as arrays
  of vertex and texture coordinates (bufobj_coords and bufobj_texcoords)

  *) initializing a buffer object by the CPU and using the data as an array of
  coordinate indices (bufobj_coordindex)

  Note that the CUDA templates representation of OpenGL buffer objects can be
  multidimensional arrays of compound data types (such as float4), although
  OpenGL interprets buffer objects as linear arrays of elementary data types.

*/

#include <iostream>
#include <stdexcept>

#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

#ifdef _WIN32
#include <GL/glew.h>
#else
#include <GL/gl.h>
#include <GL/glext.h>
#endif

#include <GL/glut.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/gilreference.hpp>
#include <cudatemplates/opengl/texture.hpp>
#include <cudatemplates/hostmemoryheap.hpp>

#include "buffer_object.hpp"

using namespace std;

#define CHECK_GL_ERRORS							\
  {									\
    GLenum err = glGetError();						\
    if (err)								\
      printf( "Error %x at line %d, %s\n", err, __LINE__, gluErrorString(err)); \
  }

#define SUBDIV_X  127
#define SUBDIV_Y  127

#define WIREFRAME 0


extern void init_geometry(BufferObjectFloat4Type &bufobj_coords, BufferObjectFloat2Type &bufobj_texcoords);
extern void init_topology(BufferObjectInt4Type &bufobj_coordindex);


void
reshape(int w, int h)
{
  glViewport(0, 0, w, h);
}

void
display()
{
  // initialize frame:
  glClearColor(1.0, 1.0, 1.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // render vertex array:
  glDrawElements(GL_QUADS, SUBDIV_X * SUBDIV_Y * 4, GL_UNSIGNED_INT, 0);

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

#ifndef _WIN32
    // read image:
    Cuda::GilReference2D<PixelType>::gil_image_t gil_image;
    boost::gil::png_read_image("ladybug.png", gil_image);  // must match PixelType!
    Cuda::GilReference2D<PixelType> image(gil_image);
#endif

    // init GLUT:
    glutInit(&argc, argv);
#ifndef _WIN32
    glutInitWindowSize(image.size[0] * 2, image.size[1] * 2);
#else
    glutInitWindowSize(512, 512);
#endif
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("OpenGL buffer object demo");
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
#ifdef _WIN32
    if (glewInit() != GLEW_OK) {
      printf("glewInit failed. Exiting...\n");
      exit(1);
    }
    CHECK_GL_ERRORS;
#endif

    // create OpenGL buffer object for image and copy data:
#ifndef _WIN32
    BufferObjectPixelType bufobj_image(image.size);
    copy(bufobj_image, image);
    // create OpenGL texture and copy data
    // (note that the image data could also be copied directly to the texture,
    // this is just to demonstrate the use of a buffer object for pixel data):
    Cuda::OpenGL::Texture<PixelType, 2> texture(image.size);
    copy(texture, bufobj_image);
#else
    BufferObjectPixelType bufobj_image(Cuda::Size<2>(256,256));
    Cuda::HostMemoryHeap2D<PixelType> h_img(Cuda::Size<2>(256,256));
    for(int i = 0; i < 256*256; i++) {
      h_img[i].x = i;
      h_img[i].y = 0;
      h_img[i].z = 0;
    }
    copy(bufobj_image, h_img);
    // create OpenGL texture and copy data
    // (note that the image data could also be copied directly to the texture,
    // this is just to demonstrate the use of a buffer object for pixel data):
    Cuda::OpenGL::Texture<PixelType, 2> texture(Cuda::Size<2>(256,256));
    copy(texture, bufobj_image);
#endif

    bufobj_image.disconnect();
    texture.bind();

    // enable textures and set parameters:
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // enable arrays:
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    // create OpenGL buffer objects for geometry and topology:
    Cuda::Size<2> size(SUBDIV_X, SUBDIV_Y), size1(SUBDIV_X + 1, SUBDIV_Y + 1);
    BufferObjectFloat4Type bufobj_coords(size1);
    BufferObjectFloat2Type bufobj_texcoords (size1);
    BufferObjectInt4Type bufobj_coordindex(size, GL_ELEMENT_ARRAY_BUFFER);

    // init buffer objects:
    init_geometry(bufobj_coords, bufobj_texcoords);
    init_topology(bufobj_coordindex);

#if WIREFRAME
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
#endif

    // enter main loop:
    glutMainLoop();
  }
  catch(const std::exception &e) {
    cerr << e.what() << endl;
    return 1;
  }

  return 0;
}
