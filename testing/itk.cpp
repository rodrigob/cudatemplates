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

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <cudatemplates/copy.hpp>
#include <cudatemplates/itkreference.hpp>
#include <cudatemplates/devicememorylinear.hpp>

using namespace std;


// prepare some ITK data types:
typedef unsigned char PixelType;
typedef itk::Image<PixelType, 2> ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;


/**
   Read ITK image.
   When the image pointer is returned by this function, the ITK file reader
   goes out of scope and decreases the reference count of the image. However,
   the image is not destructed at this time since the Cuda::ItkReference class
   holds an ImageType::Pointer to the image during its lifetime.
   @return pointer to ITK image
*/
ImageType::Pointer
read_image()
{
  // read input image:
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName("cameraman.png");
  reader->Update();
  return reader->GetOutput();
}

int
main()
{
  try {
    // create reference to ITK input image for use with CUDA classes:
    Cuda::ItkReference<PixelType, 2> ref_image_input(read_image());

    // create empty ITK output image:
    ImageType::Pointer image_output = ImageType::New();

    // create reference to ITK output image and initialize with same size as input image:
    Cuda::Spacing<2> spacing;
    spacing[0] = spacing[1] = 1;
    Cuda::ItkReference<PixelType, 2> ref_image_output(ref_image_input.size, spacing, image_output);

    // copy input image to GPU and back to output image (i.e., CPU/GPU roundtrip):
    Cuda::DeviceMemoryLinear<PixelType, 2> tmp(ref_image_input.size);
    Cuda::copy(tmp, ref_image_input);
    Cuda::copy(ref_image_output, tmp);

    // write output image:
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("cameraman_itk_out.png");
    writer->SetInput(image_output);
    writer->Update();
  }
  catch(const exception &e) {
    cerr << e.what() << endl;
    return 1;
  }

  return 0;
}
