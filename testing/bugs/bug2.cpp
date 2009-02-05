#include <stdio.h>

#include "cudatemplates/hostmemoryheap.hpp"
#include "cudatemplates/copy.hpp"

#include <math.h>

int main() {

    printf("--- is there a bug using large arrays? --- \n");

    const int num_tests = 4;
    int cols[4] = {1, 1, 16384, 16385};
    int rows[4] = {1, 32768, 1, 1};

    printf(" maximum size for array: w: %5.0f h: %5.0f \n", pow(2.0f,16), pow(2.0f,15));

    for (int test = 0; test < num_tests; test++) {    
        // init reference        
        Cuda::HostMemoryHeap<float,2> reference_hmh(Cuda::Size<2>(cols[test], rows[test]));
        for (int i = 0; i < cols[test] * rows[test]; i++)
            reference_hmh.getBuffer()[i] = rand() / 100.0f;

        // copy to array    
        Cuda::Array<float,2> array_a(reference_hmh);

        // copy back
        Cuda::HostMemoryHeap<float,2> test_hmh(array_a);
        
        // check
        double sum = 0;
        for (int i = 0; i < cols[test] * rows[test]; i++)
            sum += abs(reference_hmh.getBuffer()[i] - test_hmh.getBuffer()[i]);

        printf(" summed copy error for size w: %d h: %d = %f\n", cols[test], rows[test], sum);
    }
    return 0;
}
