#include "cudatemplates/hostmemoryheap.hpp"
#include "cudatemplates/devicememorypitched.hpp"
#include "cudatemplates/copy.hpp"


using namespace Cuda;

template <typename T>
class Blubb
{
public:
  Blubb(){
    in_h_ = NULL;
    in_d_ = NULL;
  };
  ~Blubb(){
    if (in_h_)
      delete in_h_;
    if (in_d_)
      delete in_d_;
  };

  void setInput(const HostMemoryHeap<T,2>& in){
    in_h_ = new HostMemoryHeap<T,2>(in);
  };
  void setInput(const DeviceMemoryPitched<T,2>& in){
    in_d_ = new DeviceMemoryPitched<T,2>(in);
  };

private:
  HostMemoryHeap<T,2>* in_h_;
  DeviceMemoryPitched<T,2>* in_d_;
};




int main() {
  Blubb<float> my_blubb;
  size_t my_size = 256;

  HostMemoryHeap<float,2> input_h(Size<2>(my_size,my_size));

  DeviceMemoryPitched<float,2> input_d(input_h);
  
  //my_blubb.setInput(input_h);
  my_blubb.setInput(input_d);

  return 0;
}