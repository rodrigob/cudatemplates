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

#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/gilreference.hpp>
#include <cudatemplates/devicememorylinear.hpp>

using namespace std;
using namespace boost;


typedef unsigned char PixelType;


int
main()
{
  try {
    // create gil input and output images:
    Cuda::GilReference2D<PixelType>::gil_image_t image_input, image_output;

    // read gil image:
    gil::png_read_image("cameraman.png", image_input);

    // create reference to gil input image for use with CUDA classes:
    Cuda::GilReference2D<PixelType> ref_image_input(image_input);

    // create reference to gil output image and initialize with same size as input image:
    Cuda::GilReference2D<PixelType> ref_image_output(ref_image_input.size, image_output);

    // copy input image to GPU and back to output image (i.e., CPU/GPU roundtrip):
    Cuda::DeviceMemoryLinear<PixelType, 2> tmp(ref_image_input.size);
    Cuda::copy(tmp, ref_image_input);
    Cuda::copy(ref_image_output, tmp);

    // write output image:
    gil::png_write_view("cameraman_gil_out.png", gil::view(image_output));
  }
  catch(const std::exception &e) {
    cerr << e.what() << endl;
    return 1;
  }

  return 0;
}
