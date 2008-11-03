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
#include "highgui.h"

#include <cudatemplates/copy.hpp>
#include <cudatemplates/iplreference.hpp>
#include <cudatemplates/devicememorylinear.hpp>

typedef unsigned char PixelType;

/**
   Unlike e.g. ITK, OpenCV does not take care of handling the image
   data with e.g. shared pointers and there the user has to be sure to
   delete the constructed IplImage and the cudatemplates
   representation of it in the right order.
*/

int
main()
{
  try {
    // read test input image (grayscale mode)
    IplImage* image_input = cvLoadImage( "cameraman.png", 0);

    // create reference to IplImage for use with CUDA classes:
    Cuda::IplReference<PixelType, 2> host_image_input(image_input);

    // create empty Ipl output image:
    IplImage* image_output = cvCreateImage( cvGetSize(image_input) , image_input->depth, 
					    image_input->nChannels );

    // create reference to Ipl output image and initialize with same size as input image:
    Cuda::IplReference<PixelType, 2> host_image_output(host_image_input.size, image_output);

    // copy input image to GPU and back to output image (i.e., CPU/GPU roundtrip):
    Cuda::DeviceMemoryPitched<PixelType, 2> device_image(host_image_input.size);
    Cuda::copy(device_image, host_image_input);
    Cuda::copy(host_image_output, device_image);

    // write output image:
    cvSaveImage( "cameraman_ipl_out.png", image_output );
  }
  catch(const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
