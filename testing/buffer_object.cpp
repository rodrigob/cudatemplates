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

#include <iostream>
#include <stdexcept>

#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

#include <GL/glut.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/gilreference.hpp>
#include <cudatemplates/opengl/bufferobject.hpp>
#include <cudatemplates/opengl/copy.hpp>
#include <cudatemplates/opengl/texture.hpp>

using namespace std;


#define WIREFRAME 0


typedef unsigned char PixelType;

const int SUBDIV = 32;


void
reshape(int w, int h)
{
  glViewport(0, 0, w, h);
}

void
display()
{
  glClearColor(1.0, 1.0, 1.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glDrawElements(GL_QUADS, SUBDIV * SUBDIV * 4, GL_UNSIGNED_INT, 0);

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
    // read image:
    Cuda::GilReference2D<PixelType>::gil_image_t gil_image;
    boost::gil::png_read_image("cameraman.png", gil_image);
    Cuda::GilReference2D<PixelType> image(gil_image);

    // init GLUT:
    glutInit(&argc, argv);
    glutInitWindowSize(image.size[0], image.size[1]);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("OpenGL buffer object demo");
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    // create OpenGL texture:
    // texture.alloc(image.size);
    Cuda::OpenGL::Texture<PixelType, 2> texture(image.size);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // create buffer object for image:
    Cuda::OpenGL::BufferObject<PixelType, 2> bufobj(image.size);

    // copy image to buffer object:
    copy(bufobj, image);

    // copy buffer object to texture:
    copy(texture, bufobj);

    bufobj.disconnect();
    texture.bind();

    // create coordinate array:
    GLfloat coords[(SUBDIV + 1) * (SUBDIV + 1) * 3];
    GLfloat texcoords[(SUBDIV + 1) * (SUBDIV + 1) * 2];
    GLfloat *pc = coords, *pt = texcoords;

    for(int i = 0; i <= SUBDIV; ++i)
      for(int j = 0; j <= SUBDIV; ++j) {
	float dx = (float)i / SUBDIV - 0.5;
	float dy = 0.5 - (float)j / SUBDIV;
	float d = sqrt(dx * dx + dy * dy);
	float c = (d > 0) ? 2 * pow(d, 0.2) : 0;
	*(pc++) = c * dx;
	*(pc++) = c * dy;
	*(pc++) = 0;
	*(pt++) = (float)i / SUBDIV;
	*(pt++) = (float)j / SUBDIV;
      }

    // create coordinate index array:
    int coordindex[SUBDIV * SUBDIV * 4];
    int *pi = coordindex;

    for(int i = 0; i < SUBDIV; ++i)
      for(int j = 0; j < SUBDIV; ++j) {
	int v0 = i * (SUBDIV + 1) + j;
	*(pi++) = v0;
	*(pi++) = v0 + 1;
	*(pi++) = v0 + SUBDIV + 2;
	*(pi++) = v0 + SUBDIV + 1;
      }

    // create CUDA templates references to the arrays:
    Cuda::HostMemoryReference<GLfloat, 1> ref_coords(Cuda::Size<1>((SUBDIV + 1) * (SUBDIV + 1) * 3), coords);
    Cuda::HostMemoryReference<GLfloat, 1> ref_texcoords(Cuda::Size<1>((SUBDIV + 1) * (SUBDIV + 1) * 2), texcoords);
    Cuda::HostMemoryReference<int, 1> ref_coordindex(Cuda::Size<1>(SUBDIV * SUBDIV * 4), coordindex);

    // create CUDA template OpenGL buffer objects:
    Cuda::OpenGL::BufferObject<GLfloat, 1> bufobj_coords((SUBDIV + 1) * (SUBDIV + 1) * 3);
    Cuda::OpenGL::BufferObject<GLfloat, 1> bufobj_texcoords((SUBDIV + 1) * (SUBDIV + 1) * 2);
    Cuda::OpenGL::BufferObject<int, 1> bufobj_coordindex(SUBDIV * SUBDIV * 4, GL_ELEMENT_ARRAY_BUFFER);

    // copy data to buffer objects:
    copy(bufobj_coords, ref_coords);
    copy(bufobj_texcoords, ref_texcoords);
    copy(bufobj_coordindex, ref_coordindex);

    bufobj_coords.disconnect();
    bufobj_texcoords.disconnect();
    bufobj_coordindex.disconnect();

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    bufobj_coords.bind();
    glVertexPointer(3, GL_FLOAT, 0, 0);
    bufobj_texcoords.bind();
    glTexCoordPointer(2, GL_FLOAT, 0, 0);
    bufobj_coordindex.bind();

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
