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

#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#include <cudatemplates/copy.hpp>

namespace Cuda {

  /** Image. 
   * Image representation or similar data structures on device and host memory.
   *
   * This is the base class for all kind of data structures that are
   * needed on both, the host and the device.
   *
   * The class is templeted over the type of host and device memory
   * representation and the type of elements. 
   *
   * BE AWARE that there is no possibility to create Reference types
   * of the CudaTemplates library due to the lack of partial
   * specialization of member functions (here the constructor). As
   * soon as a workaround is found the functionality will be
   * implemented. If you need a reference type by now, please build it
   * yourself.
   *
   * - HostType = CudaTemplates representation of the host memory.
   * - DeviceType = CudaTemplates representation of the device memory.
   * - PixelType = Element type stored at each posizion.
   *
   */
  template <class HostType, class DeviceType, class PixelType>
  class Image
  {
  public:

#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
    /** Default constructor. */
    inline Image() : 
      nChannels_(0),
      interleaved_(false),
      hostEntity_(NULL),
      deviceEntity_(NULL),
      imageAvailable_(false), 
      hostModified_(false),
      deviceModified_(false)
      {
      }
#endif

    /** Constructor.
     * Takes a CudaTemplates representation of host memory and
     * automatically creates the correspsonding device representation.
     * @param inputHost CudaTemplate representation of host memory
     */
    inline Image(HostType* inputHost) :
      nChannels_(0),
      interleaved_(false),
      hostModified_(true),
      deviceModified_(false)
      {
	assert(inputHost != NULL);
	hostEntity_ = inputHost;

	deviceEntity_ = new DeviceType(hostEntity_->size);
	Cuda::copy(*deviceEntity_, *hostEntity_);
	
	imageAvailable_ = true;
      }

    /** Constructor.
     * Takes a CudaTemplates representation of device memory and
     * automatically creates the correspsonding host representation.
     * @param inputDevice CudaTemplate representation of Device memory
     */
    inline Image(DeviceType* inputDevice) :
      nChannels_(0),
      interleaved_(false),
      hostModified_(false),
      deviceModified_(true)
      {
	assert(inputDevice != NULL);
	deviceEntity_ = inputDevice;

	hostEntity_ = new HostType(deviceEntity_->size);
	Cuda::copy(*hostEntity_, *deviceEntity_);

	imageAvailable_ = true;
      }


    /** Constructor.
     * Takes a CudaTemplate representation of host and device memory.
     * @param inputHost CudaTemplate representation of host memory
     * @param inputDevice CudaTemplate representation of device memory
     */
    inline Image(HostType* inputHost, DeviceType* inputDevice) :
      nChannels_(0),
      interleaved_(false),
      hostModified_(false),
      deviceModified_(false)
      {
	// Check if sizes are equal. No guarantee that data is equal
	// but at least better than nothing...
	assert(inputHost->size == inputDevice->size);
	hostEntity_ = inputHost;
	deviceEntity_ = inputDevice;
	imageAvailable_ = true;
	assert(hostEntity_ != NULL && deviceEntity_ != NULL);
      }

    virtual ~Image()
      {
//	std::cout << "Destructor of Image called" << std::endl;
	delete(hostEntity_);
	hostEntity_ = 0;
	delete(deviceEntity_);
	deviceEntity_ = 0;
      }

    /** Updates the data on the host side - also works for reference structures (data is copied!)
     * @param[in] data pixel data. (e.g. of an IplImage)
     * @param[in] width image width.
     * @param[in] height image height.
     * @param[in] widthStep row size (default = width). (e.g. IplImage::widthStep if used as OpenCV connector)
     */
    inline void updateHostBuffer(unsigned char* data, int width, int height, uint widthStep = 0)
      {
	if(data == NULL)
	  return;
	if(widthStep == 0)
	  widthStep = width;

	assert(hostEntity_ != NULL && deviceEntity_ != NULL);
	Cuda::Size<2> size = hostEntity_->size;
	// copy and norm single pixel values
	PixelType* buffer = hostEntity_->getBuffer();

	for(unsigned y = 0; y < size[1]; ++y)
	{
	  for(unsigned x = 0; x < size[0]; ++x)
	  {
	    buffer[(size[0] * y)+x] = 
	      static_cast<PixelType>(data[(widthStep * y)+x])/255.0f;
	  }
	}
	hostModified_ = true;
	this->updateDeviceEntity();
      }

    // getters --------------------------------------------------------------------

    /** Get the CudaTemplates reprensetation of the host memory.
     * The memory is automatically synchronized if the device memory
     * was modified.
     * @return templated host memory
    */
    inline HostType* getHostEntity()
      { 
	assert(hostEntity_ != NULL && deviceEntity_ != NULL);
	if(deviceModified_) updateHostEntity();
	return hostEntity_;
      }

    /** Get the CudaTemplates reprensetation of the device memory.
     * The memory is automatically synchronized if the host memory
     * was modified.
     * @return templated device memory
    */
    inline DeviceType* getDeviceEntity() 
      { 
	assert(hostEntity_ != NULL && deviceEntity_ != NULL);
	if(hostModified_) this->updateDeviceEntity();
	return deviceEntity_;
      }

    /** Get the templated pixel reprensetation of the host memory.
     * The memory is automatically synchronized if the device memory
     * was modified.
     * @return templated pixel buffer of host memory
    */
    inline PixelType*  getHostBuffer()
      { 
	assert(hostEntity_ != NULL && deviceEntity_ != NULL);
	if(deviceModified_) this->updateHostEntity();
	return hostEntity_->getBuffer();
      }
    /** Get the templated pixel reprensetation of the device memory.
     * The memory is automatically synchronized if the device memory
     * 	was modified.
     *	@return templated pixel buffer of host memory
    */
    inline PixelType* getDeviceBuffer()
      {
	assert(hostEntity_ != NULL && deviceEntity_ != NULL);
	if(hostModified_) this->updateDeviceEntity();
	return deviceEntity_->getBuffer(); 
      }

    /** Get size.
     * @return size of the stored device memory layout in each dimension.
    */
    inline Cuda::Size<2> getSize() const
      { 
	assert(hostEntity_ != NULL && deviceEntity_ != NULL);
	return deviceEntity_->size;
      }

    /** Get stride.
     * @return stride of the stored device memory layout in each dimension.
    */
    inline Cuda::Size<2> getStride() const
      { 
	assert(hostEntity_ != NULL && deviceEntity_ != NULL);
	return deviceEntity_->stride;
      }

    /** Get pitch.
     * @return number of bytes in a row (including any padding)
     */
    inline size_t getPitch() const { return deviceEntity_->getPitch(); }

    /** Get flag if an image is available.
     * @return true if a valid host and device representation is available and false if not.
     */
    inline bool imageAvailable() const { return imageAvailable_; }
    /** Get flag if host instance was modified.
     * @return true if the host memory was modified and not synchronized with the device memory.
     */
    inline bool hostModified() const { return hostModified_; }
    /** Get flag if device instance was modified.
     * @return true if the device memory was modified and not synchronized with the host memory.
    */
    inline bool deviceModified() const { return deviceModified_; }


    /** Get Number of channels.
     * @return Number of color channels.
     */
    inline uint getNChannels() const { return nChannels_; }

    /** Get flag if channels are saved interleaved.
     * @return TRUE if channels are greater 1 and saved in interleaved
     * mode. FALSE if saved in planar mode or only one channel is
     * available.
     */
    inline bool interleaved() const { return interleaved_; }

    uint nChannels_;       /**< Number of channels saved in the image structure.*/
    bool interleaved_;     /**< Flag if color channels are interleaved or planar.*/

  protected:
    
    /** Updates the host memory with the device memory */
    inline void updateHostEntity()
      {
	Cuda::copy(*hostEntity_, *deviceEntity_);
	deviceModified_ = false;
      }

    /** Updates the device memory with the host memory */
    inline void updateDeviceEntity()
      {
	Cuda::copy(*deviceEntity_, *hostEntity_);
	hostModified_ = false;
      }

    HostType* hostEntity_;     /**< CudaTemplate representation of host memory.*/
    DeviceType* deviceEntity_; /**< CudaTemplate representation of device memory.*/
    
    bool imageAvailable_; /**< Flag if host and device representations are available.*/
    bool hostModified_;   /**< Flag if host representation was modified.*/
    bool deviceModified_; /**< Flag if device representation was modified.*/

  private:

  };
}

#endif //CUDA_IMAGE_H
