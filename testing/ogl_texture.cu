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

#include <stdio.h>

#ifdef _WIN32
#include <GL/glew.h>
#else
#include <GL/gl.h>
#include <GL/glext.h>
#endif
#include <GL/glut.h>
#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/opengl/bufferobject.hpp>
#include <cudatemplates/opengl/texture.hpp>
#include <cudatemplates/opengl/copy.hpp>

#include <cutil_math.h>


const size_t TESTSIZE = 64;
int testcalls = 0;

#define CHECK_GL_ERRORS							\
  {									\
    GLenum err = glGetError();						\
    if (err)								\
      printf( "Error %x at line %d, %s\n", err, __LINE__, gluErrorString(err)); \
  }

template<class type, bool neg>
type myrand()
{
  return ((type)(rand() %  RAND_MAX))/( RAND_MAX/1024)-(neg?(type)(512):(type)(0));
}
template<class type, bool neg>
type myrand1()
{
  type x;
  x.x = myrand<float,neg>();
  return x;
}
template<class type, bool neg>
type myrand2()
{
  type x;
  x.x = myrand<float,neg>();;
  x.y = myrand<float,neg>();;
  return x;
}
template<class type, bool neg>
type myrand3()
{
  type x;
  x.x = myrand<float,neg>();;
  x.y = myrand<float,neg>();;
  x.z = myrand<float,neg>();;
  return x;
}
template<class type, bool neg>
type myrand4()
{
  type x;
  x.x = myrand<float,neg>();;
  x.y = myrand<float,neg>();;
  x.z = myrand<float,neg>();;
  x.w = myrand<float,neg>();;
  return x;
}

template<class type>
bool mycomp(const type& a, const type & b)
{
  return a == b;
}
template<class type>
bool mycomp1(const type& a, const type & b)
{
  return a.x == b.x;
}
template<class type>
bool mycomp2(const type& a, const type & b)
{
  return a.x == b.x && a.y == b.y;
}
template<class type>
bool mycomp3(const type& a, const type & b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
template<class type>
bool mycomp4(const type& a, const type & b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template<class type>
int testOglTexture(type (*randFunc)(), bool (*comp)(const type&, const type&))
{
  ++testcalls;
  Cuda::HostMemoryHeap<type, 2> h_data(Cuda::Size<2>(TESTSIZE,TESTSIZE));
  Cuda::HostMemoryHeap<type, 2> h_data2(Cuda::Size<2>(TESTSIZE,TESTSIZE));
  Cuda::OpenGL::Texture<type, 2> texture(Cuda::Size<2>(TESTSIZE,TESTSIZE));

  for(unsigned i = 0; i < TESTSIZE*TESTSIZE; i++)
    h_data[i] = randFunc();

  Cuda::OpenGL::copy(texture, h_data);

  texture.bind();
  glGetTexImage(GL_TEXTURE_2D, 0, Cuda::OpenGL::getFormat<type>(), Cuda::OpenGL::getType<type>(), h_data2.getBuffer());
  texture.unbind();


  //check
  for(size_t i = 0; i < TESTSIZE*TESTSIZE; ++i)
  {
    type a = h_data[i];
    type b = h_data2[i];
    if(!comp(a,b))
      return 1;
  }

  return 0;
}
int main(int argc, char *argv[])
{
  glutInit(&argc, argv);
  glutInitWindowSize(512, 512);

  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow("OpenGL Texture Test");
  
  #ifdef _WIN32
    if (glewInit() != GLEW_OK)
    {
      fprintf(stderr, "GLEW Init Failed\n");
      return 1;
    }
  #endif

  const GLubyte *str = glGetString(GL_EXTENSIONS);
  bool has_texture_integer = (strstr((const char *)str, "GL_EXT_texture_integer") != 0);
  bool has_texture_float = (strstr((const char *)str, "GL_ARB_texture_float") != 0);

  if(!has_texture_integer)
    fprintf(stderr, "GL_EXT_texture_integer not supported!\n");
  if(!has_texture_float)
    fprintf(stderr, "GL_ARB_texture_float not supported!\n");

  int result = 0;
  result += testOglTexture<unsigned char>(&myrand<unsigned char, true>,  &mycomp<unsigned char>);
  result += testOglTexture<struct uchar1>(&myrand1<struct uchar1, true>, &mycomp1<struct uchar1>);
  result += testOglTexture<struct uchar2>(&myrand2<struct uchar2, true>, &mycomp2<struct uchar2>);
  result += testOglTexture<struct uchar3>(&myrand3<struct uchar3, true>, &mycomp3<struct uchar3>);
  result += testOglTexture<struct uchar4>(&myrand4<struct uchar4, true>, &mycomp4<struct uchar4>);

  result += testOglTexture<        char>(&myrand<char, true>,         &mycomp< char>);
  result += testOglTexture<struct char1>(&myrand1<struct char1, true>, &mycomp1<struct char1>);
  result += testOglTexture<struct char2>(&myrand2<struct char2, true>, &mycomp2<struct char2>);
  result += testOglTexture<struct char3>(&myrand3<struct char3, true>, &mycomp3<struct char3>);
  result += testOglTexture<struct char4>(&myrand4<struct char4, true>, &mycomp4<struct char4>);

  result += testOglTexture<unsigned short>(&myrand<unsigned short, true>,  &mycomp<unsigned short>);
  result += testOglTexture<struct ushort1>(&myrand1<struct ushort1, true>, &mycomp1<struct ushort1>);
  result += testOglTexture<struct ushort2>(&myrand2<struct ushort2, true>, &mycomp2<struct ushort2>);
  result += testOglTexture<struct ushort3>(&myrand3<struct ushort3, true>, &mycomp3<struct ushort3>);
  result += testOglTexture<struct ushort4>(&myrand4<struct ushort4, true>, &mycomp4<struct ushort4>);

  result += testOglTexture<       short>(&myrand< short, true>,          &mycomp< short>);
  result += testOglTexture<struct short1>(&myrand1<struct short1, true>, &mycomp1<struct short1>);
  result += testOglTexture<struct short2>(&myrand2<struct short2, true>, &mycomp2<struct short2>);
  result += testOglTexture<struct short3>(&myrand3<struct short3, true>, &mycomp3<struct short3>);
  result += testOglTexture<struct short4>(&myrand4<struct short4, true>, &mycomp4<struct short4>);

  result += testOglTexture<unsigned int>(&myrand<unsigned int, true>,  &mycomp<unsigned int>);
  result += testOglTexture<struct uint1>(&myrand1<struct uint1, true>, &mycomp1<struct uint1>);
  result += testOglTexture<struct uint2>(&myrand2<struct uint2, true>, &mycomp2<struct uint2>);
  result += testOglTexture<struct uint3>(&myrand3<struct uint3, true>, &mycomp3<struct uint3>);
  result += testOglTexture<struct uint4>(&myrand4<struct uint4, true>, &mycomp4<struct uint4>);

  result += testOglTexture<        int>(&myrand<int, true>,          &mycomp<int>);
  result += testOglTexture<struct int1>(&myrand1<struct int1, true>, &mycomp1<struct int1>);
  result += testOglTexture<struct int2>(&myrand2<struct int2, true>, &mycomp2<struct int2>);
  result += testOglTexture<struct int3>(&myrand3<struct int3, true>, &mycomp3<struct int3>);
  result += testOglTexture<struct int4>(&myrand4<struct int4, true>, &mycomp4<struct int4>);
  

  result += testOglTexture<        float>(&myrand< float, true>,         &mycomp< float>);
  result += testOglTexture<struct float1>(&myrand1<struct float1, true>, &mycomp1<struct float1>);
  result += testOglTexture<struct float2>(&myrand2<struct float2, true>, &mycomp2<struct float2>);
  result += testOglTexture<struct float3>(&myrand3<struct float3, true>, &mycomp3<struct float3>);
  result += testOglTexture<struct float4>(&myrand4<struct float4, true>, &mycomp4<struct float4>);

  if(result)
    fprintf(stderr, "%d/%d OpenGL Texture type tests failed\n", result, testcalls);
  return result;
};
