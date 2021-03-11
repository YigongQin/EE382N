#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cmath>
#include "CycleTimer.h"
#include "histo.h"

void printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

extern float toBW(int bytes, float sec);

// edit arguments if necessary
__global__ void
histo_kernel(unsigned char *data, 
             unsigned long len_in_byte, 
             unsigned int *histo_r, 
             unsigned int *histo_g, 
             unsigned int *histo_b, 
             unsigned int bin_width) {
    // compute global index for each thread 
    // with thread ID within a thread block and thread block ID
}

// this function computes histogram and returns pointer of histogram result
unsigned int *histogramCuda(unsigned char *data, int image_len_in_byte, unsigned int num_bins) {
   
    unsigned int *histo = NULL;

    // TODO allocate array for histogram for each R,G,B

    // TODO compute bin width 

    // TODO compute number of blocks and threads per block

    // TODO compute an amount of work per each thread

    // TODO allocate device memory for image and histogram on the GPU using cudaMalloc

    // TODO copy input image array to the GPU using cudaMemcpy

    // TODO run kernel

    // TODO sync kernel

    // TODO copy histogram from GPU using cudaMemcpy

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured while ending: code=%d, %s\n", errCode, cudaGetErrorString(errCode)); exit(-1);
    }
    
    // TODO free memory buffers on the GPU

    return histo;
}

