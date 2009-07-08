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

#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>


#include <GL/glew.h>

#include <GL/glut.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/gilreference.hpp>
#include <cudatemplates/opengl/bufferobject.hpp>
#include <cudatemplates/opengl/copy.hpp>
#include <cudatemplates/opengl/texture.hpp>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/devicememorylinear.hpp>

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
typedef float PixelType;

extern void find_crazyAddress(Cuda::HostMemoryHeap2D<PixelType> pattern, 
                             Cuda::Size<2> size, 
                             Cuda::HostMemoryHeap1D<unsigned long long> mcth, 
                             Cuda::DeviceMemoryLinear1D<unsigned long long> mct
                             );

Cuda::HostMemoryHeap2D<PixelType> h_img;
Cuda::OpenGL::BufferObject2D<PixelType> bufobj_image;
Cuda::OpenGL::Texture<PixelType, 2> texture;
Cuda::DeviceMemoryLinear1D<unsigned long long > mct(Cuda::Size<1>(2));
Cuda::HostMemoryHeap1D<unsigned long long> mcth(Cuda::Size<1>(2));

const int SIZE = 256;

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
  
  texture.bind();
  glPushAttrib(GL_ENABLE_BIT);
  // enable textures and set parameters:
  glEnable(GL_TEXTURE_2D);
  glDisable(GL_DEPTH_TEST);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glBegin(GL_QUADS);
  glTexCoord2f(0, 1); glColor3d(1.0, 0.0, 0.0); glVertex3f(-1, -1, 0);
  glTexCoord2f(1, 1); glColor3d(0.0, 1.0, 0.0); glVertex3f( 1, -1, 0);
  glTexCoord2f(1, 0); glColor3d(0.0, 0.0, 1.0); glVertex3f( 1,  1, 0);
  glTexCoord2f(0, 0); glVertex3f(-1,  1, 0);
  glEnd();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glPopAttrib();

  texture.unbind();
  CHECK_GL_ERRORS;

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
    // read image:
    Cuda::GilReference2D<PixelType>::gil_image_t gil_image;
    boost::gil::png_read_and_convert_image("ladybug.png", gil_image);  // must match PixelType!
    Cuda::GilReference2D<PixelType> image(gil_image);

    printf("image size %lu x %lu\n", image.size[0], image.size[1]);
    // init GLUT:
    glutInit(&argc, argv);
    glutInitWindowSize(image.size[0] * 1, image.size[1] * 1);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("OpenGL buffer object demo");
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    if (glewInit() != GLEW_OK) {
      printf("glewInit failed. Exiting...\n");
      exit(1);
    }
    CHECK_GL_ERRORS;


    // create OpenGL buffer object for image and copy data:
    bufobj_image.alloc(Cuda::Size<2>(image.size[0], image.size[1]));
    texture.alloc(Cuda::Size<2>(image.size[0], image.size[1]));
    h_img.alloc(Cuda::Size<2>(image.size[0], image.size[1]));
    srand ( time(NULL) );
    for(int i = 0; i < image.size[0]*image.size[1]; i++) {
      h_img[i] = rand()/(float)RAND_MAX;
    }
    copy(bufobj_image, image);
    //copy(h_img, image);

    // create OpenGL texture and copy data
    // (note that the image data could also be copied directly to the texture,
    // this is just to demonstrate the use of a buffer object for pixel data):
    
    copy(texture, bufobj_image);
    printf("image size %lu x %lu\n", image.size[0], image.size[1]);    
    find_crazyAddress(bufobj_image, Cuda::Size<2>(image.size[0], image.size[1]),mcth, mct);



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

//========================================================================
// End of file
//========================================================================
// Local Variables:
// mode: c++
// c-basic-offset: 2
// eval: (c-set-offset 'substatement-open 0)
// eval: (c-set-offset 'case-label '+)
// eval: (c-set-offset 'statement 'c-lineup-runin-statements)
// eval: (setq indent-tabs-mode nil)
// End:
//========================================================================
