#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from dev_offsetition of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
    // if(index < 4){
    //   printf("check: x = %f\n", x[index]);
    //   printf("check: y = %f\n", y[index]);
    //   printf("check: result = %f\n", result[index]);
    // }
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
  // TODO: implement and use this interface if necessary  
  return 0;
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

    float *device_x;
    float *device_y;
    float *device_result;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    // Added
    size_t array_size = total_elems * sizeof(float);
    cudaMalloc((void**)&device_x, array_size);
    cudaMalloc((void**)&device_y, array_size);
    cudaMalloc((void**)&device_result, array_size);

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    //
    // TODO: Compute number of thread blocks.
    // Added
    dim3 blockSize(threadsPerBlock);
    dim3 gridSize ((total_elems + blockSize.x - 1) / blockSize.x); // impl ceil

    double startDataCopyH2DTime = CycleTimer::currentSeconds();

    //
    // TODO: copy input arrays to the GPU using cudaMemcpy
    // Added
    cudaMemcpy((void*)device_x, (void*)xarray, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)device_y, (void*)yarray, array_size, cudaMemcpyHostToDevice);

    // Compute time of coping data from CPU to GPU
    double endDataCopyH2DTime = CycleTimer::currentSeconds();
    double timeH2D = - startDataCopyH2DTime + endDataCopyH2DTime;
    timeCopyH2DAvg += timeH2D;

    //
    // TODO: insert time here to begin timing only the kernel
    // Did
    double startGPUTime = CycleTimer::currentSeconds();

    // run saxpy_kernel on the GPU
    saxpy_kernel<<<gridSize,blockSize>>>(total_elems, alpha, device_x, device_y, device_result);

    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaDeviceSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    cudaDeviceSynchronize();

    double endGPUTime = CycleTimer::currentSeconds();
    double timeKernel = endGPUTime - startGPUTime;
    timeKernelAvg += timeKernel;
    
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    
    double startDataCopyD2HTime = CycleTimer::currentSeconds();

    //
    // TODO: copy result from GPU using cudaMemcpy
    // Added
    cudaMemcpy(resultarray, device_result, array_size, cudaMemcpyDeviceToHost);

    // Compute time of coping data from CPU to GPU
    double endDataCopyD2HTime = CycleTimer::currentSeconds();
    double timeD2H = - startDataCopyD2HTime + endDataCopyD2HTime;
    timeCopyD2HAvg += timeD2H;

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;

    //
    // TODO free memory buffers on the GPU
    // Added
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
