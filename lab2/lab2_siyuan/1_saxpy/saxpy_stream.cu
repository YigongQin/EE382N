#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
  // TODO: implement and use this interface if necessary  
}

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
}

void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 512; // change this if necessary
    int stream_size = (total_elems+partitions-1)/partitions;

    float *device_x;
    float *device_y;
    float *device_result;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    cudaMalloc(&device_x, total_elems*sizeof(float));
    cudaMalloc(&device_y, total_elems*sizeof(float));
    cudaMalloc(&device_result, total_elems*sizeof(float));
    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();
    cudaStream_t stream[partitions];
    for (int i = 0; i < partitions; ++i){
        cudaStreamCreate(&stream[i]);
    }
    double timeStamp0[partitions];
    double timeStamp1[partitions];
    double timeStamp2[partitions];
    double timeStamp3[partitions];
    for (int i=0; i<partitions; i++) {
  
        //
        // TODO: copy input arrays to the GPU using cudaMemcpy
        //
        int offset = i * stream_size;
        timeStamp0[i] = CycleTimer::currentSeconds();
        cudaMemcpyAsync(&device_x[offset], &xarray[offset], stream_size*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&device_y[offset], &yarray[offset], stream_size*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        //
        // TODO: insert time here to begin timing only the kernel
        //
        timeStamp1[i] = CycleTimer::currentSeconds();
        // compute number of blocks and threads per block
        int numBlocks = (stream_size+threadsPerBlock-1)/threadsPerBlock;
        // run saxpy_kernel on the GPU
        saxpy_kernel<<<numBlocks, threadsPerBlock, 0, stream[i]>>>(stream_size, alpha, &device_x[offset], &device_y[offset], &device_result[offset]);
        //
        // TODO: insert timer here to time only the kernel.  Since the
        // kernel will run asynchronously with the calling CPU thread, you
        // need to call cudaDeviceSynchronize() before your timer to
        // ensure the kernel running on the GPU has completed.  (Otherwise
        // you will incorrectly observe that almost no time elapses!)
        //
        // cudaDeviceSynchronize();
        timeStamp2[i] =  CycleTimer::currentSeconds();
        cudaError_t errCode = cudaPeekAtLastError();
        if (errCode != cudaSuccess) {
            fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        }
    
        //
        // TODO: copy result from GPU using cudaMemcpy
        //
        cudaMemcpyAsync(&resultarray[offset], &device_result[offset], stream_size*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
        timeStamp3[i] = CycleTimer::currentSeconds();
      }
    cudaDeviceSynchronize();

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg += overallDuration;
    for(int i = 0; i<partitions; i++){
      timeCopyH2DAvg += timeStamp1[i]-timeStamp0[i];
      timeKernelAvg += timeStamp2[i]-timeStamp1[i];
      timeCopyD2HAvg += timeStamp3[i]-timeStamp2[i];
    }
    for(int i = 0; i<partitions; i++){
      printf("%f ", timeStamp0[i]);
    }
    printf("1 \n");
    for(int i = 0; i<partitions; i++){
      printf("%f ", timeStamp1[i]);
    }
    printf("2 \n");
    for(int i = 0; i<partitions; i++){
      printf("%f ", timeStamp2[i]); 
    }
    printf("3 \n");
    for(int i = 0; i<partitions; i++){
      printf("%f ", timeStamp3[i]); 
    }
    printf("\n");
    

    //
    // TODO free memory buffers on the GPU
    //
    for (int i = 0; i < partitions; i++){
      cudaStreamDestroy(stream[i]);
    }
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void
printCudaInfo() {

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
