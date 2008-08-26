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
   * \namespace Cuda
   */
  template <class HostType, class DeviceType, class PixelType>
  class Image
  {
  public:

#ifndef CUDA_NO_DEFAULT_CONSTRUCTORS
    /** Default constructor. */
    inline Image() : 
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
      hostModified_(false),
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
    Image(DeviceType* inputDevice) :
      hostModified_(false),
      deviceModified_(false)     
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
      hostModified_(false),
      deviceModified_(false)     
      {
	assert(inputHost->size == inputDevice->size);
	hostEntity_ = inputHost;
	deviceEntity_ = inputDevice;
	imageAvailable_ = true;

      }

    // getters --------------------------------------------------------------------

    /** Get the CudaTemplates reprensetation of the host memory.
     * The memory is automatically synchronized if the device memory
     * was modified.
     * @return templated host memory
    */
    inline HostType* getHostEntity()
      { 
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
	if(hostModified_) updateDeviceEntity();
	return deviceEntity_;
      }

    /** Get the templated pixel reprensetation of the host memory.
     * The memory is automatically synchronized if the device memory
     * was modified.
     * @return templated pixel buffer of host memory
    */
    inline PixelType*  getHostBuffer() const { return hostEntity_->getBuffer(); }
    /** Get the templated pixel reprensetation of the host memory.
     * The memory is automatically synchronized if the device memory
     * 	was modified.
     *	@return templated pixel buffer of host memory
    */
    inline PixelType*  getDeviceBuffer()const { return deviceEntity_->getBuffer(); }

    /** Get size.
     * @return size of the stored host and device memory layout in each dimension.
    */
    inline Cuda::Size<2> getSize() const { return deviceEntity_->size; }
    /** Get pitch.
     * @return number of bytes in a row (including any padding)
     */
    inline size_t getPitch() const { return deviceEntity_->getPitch(); }

    /** Get flag if an image is available.
     * @return true if a valid host and device representation is available and false if not.
     */
    inline bool imageAvailable() const { return imageAvailable_; };
    /** Get flag if host instance was modified.
     * @return true if the host memory was modified and not synchronized with the device memory.
     */
    inline bool hostModified() const { return hostModified_; };
    /** Get flag if device instance was modified.
     * @return true if the device memory was modified and not synchronized with the host memory.
    */
    inline bool deviceModified() const { return deviceModified_; };

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

  private:
    HostType* hostEntity_;     /**< CudaTemplate representation of host memory.*/
    DeviceType* deviceEntity_; /**< CudaTemplate representation of device memory.*/

    bool imageAvailable_; /**< Flag if host and device representations are available.*/
    bool hostModified_;   /**< Flag if host representation was modified.*/
    bool deviceModified_; /**< Flag if device representation was modified.*/
  };
}

#endif //CUDA_IMAGE_H
