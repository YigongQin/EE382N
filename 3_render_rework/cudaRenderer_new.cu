#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include "CycleTimer.h"

// add thrust
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/execution_policy.h>

// #define array_type short
#define array_type unsigned short
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
__managed__ int num_ones;

/* Helper function to round up to a power of 2. 
 */
static inline long long int nextPow2(long long int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // This conditional is in the inner loop, but it evaluates the
    // same direction for all threads so it's cost is not so
    // bad. Attempting to hoist this conditional is not a required
    // student optimization in Assignment 2
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
/*
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}
*/
////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    // dim3 blockDim(32, 32, 1);

    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}


//-----------------above haven't changed

__global__ void
incl_sweep_up(int N, int dim, int twod, int twod1, array_type* output) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N && ( (index/dim) % twod1 ==0) )
       {output[index+ dim*(twod1 -1)] += output[index+ dim*(twod -1)];}
}

__global__ void
incl_sweep_down(int N, int dim, int twod, int twod1, array_type* output) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ( (dim-1< index) && (index < N) && ( (index/dim) % twod1 ==0) ){
         output[index+ dim*(twod-1)] += output[index- dim*1];}
}

//--- the above should be correct

__global__ void
obtain_seperator(int total_size, int num_circ, int num_circ_true, int num_boxes, array_type* circ_cover_flag, int* separators, int partitionId, int partitionNum){

     //
     int index = blockIdx.x * blockDim.x + threadIdx.x;
    //  int circleid = index/num_boxes + partitionId * partitionNum;
     //update the separators by the way
     if (index<num_boxes) {separators[index]=circ_cover_flag[(num_circ_true-1)*num_boxes+index];}
                           // printf(" separa %d, loca %d", separators[index], (num_circ_true-1)*num_boxes+index);}
}

__global__ void
concurrent_write_ids(int total_size, int num_circ, int num_circ_true, int num_boxes, array_type* circ_cover_flag, int* circ_cover_id, int* separators, int partitionId, int partitionNum){

     //
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     int circleid = index/num_boxes;
     int blockid = index%num_boxes; //  index-num_boxes*circleid;
     if (index<total_size){
         if (circleid==0){
               if (circ_cover_flag[index]==1){
                   int new_loc = num_circ_true*blockid;
                   //printf("index %d, new_loc %d", index, new_loc);
                   circ_cover_id[new_loc]=0;}}
         else{
         //   if (circleid>0){
               if ( circ_cover_flag[index] - circ_cover_flag[index-num_boxes] ==1){
                int new_loc = blockid*num_circ_true +circ_cover_flag[index] -1;
                //if (circ_cover_flag[index]==2){ }
                   circ_cover_id[new_loc] = circleid + partitionId * partitionNum; }
             }
     }
     //update the separators by the way
     if (index<num_boxes) {separators[index]=circ_cover_flag[(num_circ_true-1)*num_boxes+index];}
                           // printf(" separa %d, loca %d", separators[index], (num_circ_true-1)*num_boxes+index);}
}

__global__ void
concurrent_write_ids_v2(int total_size, int num_circ, int num_circ_true, int num_box_max, int num_boxes, array_type* circ_cover_flag, int* circ_cover_id, int partitionId, int partitionNum){

     //
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     int circleid = index/num_boxes;
     int blockid = index%num_boxes; //  index-num_boxes*circleid;
     if (index<total_size){
         if (circleid==0){
               if (circ_cover_flag[index]==1){
                   int new_loc = num_box_max*blockid;
                   //printf("index %d, new_loc %d", index, new_loc);
                   circ_cover_id[new_loc]=0;}}
         else{
         //   if (circleid>0){
               if ( circ_cover_flag[index] ==1){
                int new_loc = blockid*num_box_max +circ_cover_flag[index] -1;
                //if (circ_cover_flag[index]==2){ }
                   circ_cover_id[new_loc] = circleid + partitionId * partitionNum; }
             }
     }
    //  again? need sort? is this efficient? sort: Pair: box_id, circle_id?
}

void multi_dim_inclusive_scan(int N, int lens, int dim, array_type* device_result){

    int blocksize = 512;
    int num_blocks = (N+blocksize-1)/blocksize;
    // printf("N=%d,block size = %d, number of blocks %d \n",N,blocksize,num_blocks); 
    for (int twod =1; twod <lens; twod *=2){
        int twod1 = twod*2;
            incl_sweep_up<<< num_blocks, blocksize >>>(N, dim, twod, twod1, device_result);        
    }
    for (int twod = lens/4; twod >=1; twod /=2){
        int twod1 = twod*2;
            incl_sweep_down<<< num_blocks, blocksize  >>>(N, dim, twod, twod1, device_result);
    }

}


#include "circleBoxTest.cu_inl"
__global__ void findCircsInBlock(array_type* circ_cover_flag, int num_total_blocks, int num_blockx, int num_blocky, int partitionId, int partitionNum) {
    // step1: find the circle idx and find the block idx
    int Idx = blockDim.x * blockIdx.x + threadIdx.x; // B*numCircles
    if (Idx>= partitionNum*num_total_blocks) {return;}
    int circleId = Idx / num_total_blocks + partitionId * partitionNum; //obtain the circle Id
    // int circleId = Idx / num_total_blocks; //obtain the circle Id
    int blockId  = Idx % num_total_blocks; //obtain the block Id
    // step2: justify whether this circle is in this block
    // can we use circlesBoxTest?
    
    //step2.1 obtain the block size
    // image params
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // compute the size of block
    
    int blockId_dimX = blockId % num_blockx;
    int blockId_dimY = blockId / num_blockx;

    short blockMinX = BLOCK_DIM_X * blockId_dimX;
    short blockMaxX = BLOCK_DIM_X * (blockId_dimX + 1);
    short blockMinY = BLOCK_DIM_Y * blockId_dimY;
    short blockMaxY = BLOCK_DIM_Y * (blockId_dimY + 1);  

    float blockL = blockMinX * invWidth;
    float blockR = blockMaxX * invWidth;
    float blockB = blockMinY * invHeight;
    float blockT = blockMaxY * invHeight;

    //step2.2 obtain the circle size
    int index3 = 3 * circleId;
    // read postion and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[circleId];
    // use the circleInBoxConservative in circleBoxTest.cu_inl
    if( circleInBoxConservative(p.x, p.y, rad, blockL, blockR, blockT, blockB) ){
        circ_cover_flag[Idx] = 1;
    }
    else{
        circ_cover_flag[Idx] = 0;
    }
    __syncthreads();
}

// #include "circleBoxTest.cu_inl"
__global__ void findNumCircsInBlock(int* separators, int num_total_blocks, int num_blockx, int num_blocky, int numPartitions) {
    // Aim to find separators not via multi_dim_inclusive_scan
    // check sharedMem at https://www.cnblogs.com/xiaoxiaoyibu/p/11402607.html ; to optimize memory access
    __shared__ int numCirclesPerPixel[BLOCK_DIM_X * BLOCK_DIM_Y];
    
    int numPixels = BLOCK_DIM_X * BLOCK_DIM_Y;
    int blockId = blockIdx.y * num_blockx + blockIdx.x;
    if (blockId >= num_total_blocks){return;}
    int pixelId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    
    //step2.1 obtain the block size
    // image params
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // compute the size of block
    int blockId_dimX = blockId % num_blockx;
    int blockId_dimY = blockId / num_blockx;

    short blockMinX = BLOCK_DIM_X * blockId_dimX;
    short blockMaxX = BLOCK_DIM_X * (blockId_dimX + 1);
    short blockMinY = BLOCK_DIM_Y * blockId_dimY;
    short blockMaxY = BLOCK_DIM_Y * (blockId_dimY + 1);  

    float blockL = blockMinX * invWidth;
    float blockR = blockMaxX * invWidth;
    float blockB = blockMinY * invHeight;
    float blockT = blockMaxY * invHeight;

    //step2.2 obtain the circle size
    //Each thread would take responsibility for partition of Circles
    int numCirclesPerPartition = (cuConstRendererParams.numCircles + numPartitions - 1) / numPartitions;
    // obtain the start and end
    int start = numCirclesPerPartition * pixelId;
    int end   = numCirclesPerPartition * (pixelId+1);
    if (pixelId == (numPixels - 1)){
        end = cuConstRendererParams.numCircles;
    }
    
    int numCirclesInBlockPartition = 0;
    // To find whether they are in this block and update separators[blockId]
    // How to do???
    for (int i = start; i <end; i++){
        if (i >= cuConstRendererParams.numCircles){return;}
        int index3 = 3 * i;
        // read postion and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float  rad = cuConstRendererParams.radius[i];
        // use the circleInBoxConservative in circleBoxTest.cu_inl
        if( circleInBoxConservative(p.x, p.y, rad, blockL, blockR, blockT, blockB) ){
            numCirclesInBlockPartition += 1;
        }
    }

    // such that we can have in each thread how many circles are in this block
    // then we do what? we want to sum up of the numCirclesInBlockPartition
    numCirclesPerPixel[pixelId] = numCirclesInBlockPartition;
    __syncthreads();

    // parallel reduction
    // https://zhuanlan.zhihu.com/p/41151532
    for (unsigned int j = numPixels / 2; j > 0; j >>= 1)
    {
        if (pixelId < j)
        numCirclesPerPixel[pixelId] += numCirclesPerPixel[pixelId + j];
        __syncthreads();
    }
    if (pixelId == 0)
        separators[blockId] = numCirclesPerPixel[0];
}

__global__ void findCircCoverIdInBlock(int* separators, int* circ_cover_flag, int num_total_blocks, int num_blockx, int num_blocky, int numPartitions) {
    // Aim to find circ_cover_id not via multi_dim_inclusive_scan; by the aid of separators
    // check sharedMem at https://www.cnblogs.com/xiaoxiaoyibu/p/11402607.html ; to optimize memory access
    __shared__ int numCirclesPerPixel[BLOCK_DIM_X * BLOCK_DIM_Y];
    int numPixels = BLOCK_DIM_X * BLOCK_DIM_Y;
    int blockId = blockIdx.y * num_blockx + blockIdx.x;
    if (blockId >= num_total_blocks){return;}
    int pixelId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    int numCirclesWithinThisBlock = 0;
    // obtain #circle within this block
    if (blockId == 0){numCirclesWithinThisBlock = separators[blockId];}
    else{numCirclesWithinThisBlock = separators[blockId];}
    // initialize a shared mem with size numCirclesWithinThisBlock
    // __shared__ int circCoverIdThisBlock[numCirclesWithinThisBlock];
    extern __shared__ int circCoverIdThisBlock[];

    // still like findNumCircsInBlock but we need to sort the circle id

    //step2.1 obtain the block size
    // image params
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // compute the size of block
    int blockId_dimX = blockId % num_blockx;
    int blockId_dimY = blockId / num_blockx;

    short blockMinX = BLOCK_DIM_X * blockId_dimX;
    short blockMaxX = BLOCK_DIM_X * (blockId_dimX + 1);
    short blockMinY = BLOCK_DIM_Y * blockId_dimY;
    short blockMaxY = BLOCK_DIM_Y * (blockId_dimY + 1);  

    float blockL = blockMinX * invWidth;
    float blockR = blockMaxX * invWidth;
    float blockB = blockMinY * invHeight;
    float blockT = blockMaxY * invHeight;

    //step2.2 obtain the circle size
    //Each thread would take responsibility for partition of Circles
    int numCirclesPerPartition = (cuConstRendererParams.numCircles + numPartitions - 1) / numPartitions;
    // obtain the start and end
    int start = numCirclesPerPartition * pixelId;
    int end   = numCirclesPerPartition * (pixelId+1);
    if (pixelId == (BLOCK_DIM_X * BLOCK_DIM_Y - 1)){
        end = cuConstRendererParams.numCircles;
    }
    
    int numCirclesInBlockPartition = 0;
    // To find whether they are in this block and update separators[blockId]
    // How to do???
    for (int i = start; i < end; i++){
        if (i >= cuConstRendererParams.numCircles){return;}
        int index3 = 3 * i;
        // read postion and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float  rad = cuConstRendererParams.radius[i];
        // use the circleInBoxConservative in circleBoxTest.cu_inl
        if( circleInBox(p.x, p.y, rad, blockL, blockR, blockT, blockB) ){
            numCirclesInBlockPartition += 1;
        }
    }

    // such that we can have in each thread how many circles are in this block
    // then we do what? we want to sum up of the numCirclesInBlockPartition
    numCirclesPerPixel[pixelId] = numCirclesInBlockPartition;
    __syncthreads();

    // parallel reduction
    // https://zhuanlan.zhihu.com/p/41151532
    for (unsigned int j = numPixels / 2; j > 0; j >>= 1)
    {
        if (pixelId < j)
        numCirclesPerPixel[pixelId] += numCirclesPerPixel[pixelId + j];
        __syncthreads();
    }
    if (pixelId == 0)
        separators[blockId] = numCirclesPerPixel[0];
}

__global__ void kernelRenderCircles_shared_mem(int* separators, int num_total_blocks, int num_blockx, int num_blocky, int numPartitions) {
    // Use partition to seperate numCircles can not fully parallel due to multiDimScan
    // Use sharedMem to optimize memory access
    __shared__ int numCirclesPerPixel[BLOCK_DIM_X * BLOCK_DIM_Y];
    
    int numPixels = BLOCK_DIM_X * BLOCK_DIM_Y;
    int blockId = blockIdx.y * num_blockx + blockIdx.x;
    if (blockId >= num_total_blocks){return;}
    int pixelId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    
    //step2.1 obtain the block size
    // image params
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // compute the size of block
    int blockId_dimX = blockId % num_blockx;
    int blockId_dimY = blockId / num_blockx;

    short blockMinX = BLOCK_DIM_X * blockId_dimX;
    short blockMaxX = BLOCK_DIM_X * (blockId_dimX + 1);
    short blockMinY = BLOCK_DIM_Y * blockId_dimY;
    short blockMaxY = BLOCK_DIM_Y * (blockId_dimY + 1);  

    float blockL = blockMinX * invWidth;
    float blockR = blockMaxX * invWidth;
    float blockB = blockMinY * invHeight;
    float blockT = blockMaxY * invHeight;

    //step2.2 obtain the circle size
    //Each thread would take responsibility for partition of Circles
    int numCirclesPerPartition = (cuConstRendererParams.numCircles + numPartitions - 1) / numPartitions;
    // obtain the start and end
    int start = numCirclesPerPartition * pixelId;
    int end   = numCirclesPerPartition * (pixelId+1);
    if (pixelId == (numPixels - 1)){
        end = cuConstRendererParams.numCircles;
    }
    
    int numCirclesInBlockPartition = 0;
    // To find whether they are in this block and update separators[blockId]
    // Add local recorder to record the cover_circ_id
    int * circ_cover_id_p = new int [numCirclesPerPartition];

    for (int i = start; i < end; i++){
        if (i >= cuConstRendererParams.numCircles){return;}
        int index3 = 3 * i;
        // read postion and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float  rad = cuConstRendererParams.radius[i];
        // use the circleInBoxConservative in circleBoxTest.cu_inl
        if( circleInBox(p.x, p.y, rad, blockL, blockR, blockT, blockB) ){
            numCirclesInBlockPartition += 1;
            circ_cover_id_p[numCirclesInBlockPartition] = i;
        }
    }

    // such that we can have in each thread how many circles are in this block
    // then we do what? we want to sum up of the numCirclesInBlockPartition
    numCirclesPerPixel[pixelId] = numCirclesInBlockPartition;
    __syncthreads();

    // TODO: we need a inclusive scan here and update separators! we can check the seperators


    separators[blockId] = numCirclesPerPixel[numPixels - 1];
    int totalCircles = numCirclesPerPixel[numPixels - 1];
    // update block-wise circ_cover_id here
    __shared__ int circ_cover_id_b[2500]; // 2500 is enough for circleInBox()
    
    int startAddr = 0;
    if (pixelId != 0) {startAddr = numCirclesPerPixel[pixelId - 1];}

    // how to update? AT! __syncthreads();
    for (int i =0; i < numCirclesInBlockPartition; i++){
        circ_cover_id_b[i + startAddr] = circ_cover_id_p[i];
    }
    __syncthreads();
    // parallel reduction
    // no need for parallel reduction use a inclusive scan is enough
    /*
    for (unsigned int j = numPixels / 2; j > 0; j >>= 1)
    {
        if (pixelId < j)
        numCirclesPerPixel[pixelId] += numCirclesPerPixel[pixelId + j];
        __syncthreads();
    }
    if (pixelId == 0)
        separators[blockId] = numCirclesPerPixel[0];
    */

    // directly render is okay! we donn't need another render?
    // pixel data
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelY >= imageHeight || pixelX >= imageWidth) return;

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);

    for (int i = 0; i < totalCircles; i ++){
        int circleIdx = circ_cover_id_b[i];
        int index3 = circleIdx * 3;
        // read postion and radius then use shadePixel to update
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        shadePixel(circleIdx, pixelCenterNorm, p, imgPtr);
    }
    
}

/*
void debug_set1(){
    int* debug_flag=  new int[20]; 
    int debug[20]  = {1,1,0,0,1, 0,0,1,0,0, 0,1,1,0,0, 0,0,0,1,1};
    memmove(debug_flag, debug, 20*sizeof(int));
    int* debug_flag_result = new int[20];
    int* debug_id_result = new int[20];
    for (int i = 0; i < 20; i++){
        printf("%d ", debug_flag[i]);
    }
    printf("\n");
    int* device_flag;
    int* device_id;
    int* device_separat;
    int N_rd = nextPow2(4);
    int B = 5;
    int* debug_separators = new int[B];
    int total = N_rd*B;
    printf("total %d \n",total);
    cudaMalloc((void **)&device_flag, sizeof(int) * total);
    cudaMalloc((void **)&device_id, sizeof(int) * total);
    cudaMalloc((void **)&device_separat, sizeof(int) * B);

    cudaMemcpy(device_flag, debug_flag, total * sizeof(int),cudaMemcpyHostToDevice);
    multi_dim_inclusive_scan(total, N_rd, B, device_flag);
    concurrent_write_ids<<<10,10>>>(total, N_rd, 4, B, device_flag,  device_id, device_separat);

    cudaMemcpy(debug_flag_result, device_flag, total * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_id_result, device_id, total * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_separators, device_separat, B * sizeof(int),cudaMemcpyDeviceToHost);
    //cudaMemcpy(debug_id_result, device_id, total * sizeof(int),cudaMemcpyDeviceToHost);
    //print
    for (int i = 0; i < 20; i++){
        printf("%d ", debug_flag_result[i]);
    }
    printf("\n");
    for (int i = 0; i < 20; i++){
        printf("%d ", debug_id_result[i]);
    }
    printf("\n");
    for (int i = 0; i < B; i++){
        printf("%d ", debug_separators[i]);
    }
    printf("\n");

}

void debug_set2(){
    int* debug_flag=  new int[20];
    int debug[20]  = {1,1,0,0,1, 0,0,1,0,0, 0,1,1,0,0, 0,0,0,1,1};
    memmove(debug_flag, debug, 20*sizeof(int));
    int* debug_flag_result = new int[20];
    int* debug_id_result = new int[20];
    for (int i = 0; i < 20; i++){
        printf("%d ", debug_flag[i]);
    }
    printf("\n");
    int* device_flag;
    int* device_id;
    int* device_separat;
    int N=5;
    int N_rd = nextPow2(N);
    int B = 4;
    int* debug_separators = new int[B];
    int total = N_rd*B;
    int total_rd = N_rd*B;
    printf("total %d \n",total_rd);
    cudaMalloc((void **)&device_flag, sizeof(int) * total_rd);
    cudaMalloc((void **)&device_id, sizeof(int) * total_rd);
    cudaMalloc((void **)&device_separat, sizeof(int) * B);

    cudaMemcpy(device_flag, debug_flag, total * sizeof(int),cudaMemcpyHostToDevice);
    multi_dim_inclusive_scan(total_rd, N_rd, B, device_flag);
    concurrent_write_ids<<<10,10>>>(total, N_rd, N, B, device_flag,  device_id, device_separat);

    cudaMemcpy(debug_flag_result, device_flag, total * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_id_result, device_id, total_rd * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_separators, device_separat, B * sizeof(int),cudaMemcpyDeviceToHost);
    //cudaMemcpy(debug_id_result, device_id, total * sizeof(int),cudaMemcpyDeviceToHost);
    //print
    for (int i = 0; i < 20; i++){
        printf("%d ", debug_flag_result[i]);
    }
    printf("\n");
    for (int i = 0; i < 20; i++){
        printf("%d ", debug_id_result[i]);
    }
    printf("\n");
    for (int i = 0; i < B; i++){
        printf("%d ", debug_separators[i]);
    }
    printf("\n");

}
*/
// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles(int* seperators, int* circ_cover_id, int num_blockx, int num_blocky, int numCircles) {

    // obtain block id
    int blockId = blockIdx.y * num_blockx + blockIdx.x;
    // obtain start circle and end circle using the seperators
    // int startCirc = seperators[blockId];
    // int endCirc = seperators[blockId+1];
    // int numCircsForCurrentBlock = endCirc - startCirc;
    int numCircsForCurrentBlock = seperators[blockId];
    
    // we can access the circle id through the circ_cover_id array: N*B
    int startAddInCoverId = numCircles * blockId;
    // startAddInCoverId + numCircForCurrentBlock

    // update all the pixels within this blockId
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelY >= imageHeight || pixelX >= imageWidth) return;

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    // iterate over all the circles on this block
    // AT: update by order
    for (int idx = 0; idx < numCircsForCurrentBlock; idx++){
        int circleIdx = circ_cover_id[startAddInCoverId + idx];
       // if ( (threadIdx.x==0) && (threadIdx.y==0))  {printf("%d %d  ",blockId,circleIdx);}
        int index3 = circleIdx * 3;
        // read postion and radius then use shadePixel to update
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        shadePixel(circleIdx, pixelCenterNorm, p, imgPtr);
    }
}


void
CudaRenderer::render() {

    // 256 threads per block is a healthy number
//    dim3 blockDim(256, 1);
//    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    //debug_set1();
    //debug_set2();
    int block_dimx = 32;
    int block_dimy = 32;
    int num_blockx = (image->width+block_dimx-1)/block_dimx;
    int num_blocky = (image->height+block_dimy-1)/block_dimy;    
    int num_total_blocks = num_blockx*num_blocky;
    int* circ_cover_id;
    int* separators; // size:num_total_blocks     [num_circ per block]
    array_type* circ_cover_flag; // the most big array [0,1]

    // simplified version of render
    // define dim for block
    // dim3 blockDimBlock(block_dimx, block_dimy);
    // dim3 gridDimBlock(num_blockx, num_blocky);

    // allocate separators to check whether we are right
    // cudaMalloc((void **)&separators, sizeof(int) * num_total_blocks);
    // kernelRenderCircles_shared_mem<<<gridDimBlock, blockDimBlock>>>(separators, num_total_blocks, num_blockx, num_blocky, num_blockx*num_blocky);
    // cudaDeviceSynchronize();

    if (numCircles < 10000){
        int num_circ_rd = nextPow2(numCircles); //rounded numCircles
        long total_size = numCircles*num_total_blocks;
        long total_size_rd = num_circ_rd*num_total_blocks;

        //int* check_ids = new int[total_size];
        int* check_sps = new int[num_total_blocks];
        //int* check_flags = new int[total_size_rd];
        // long total_size = numCircles;
        //int* circ_loca_ids; // concatenation of num_total_blocks variable-size arrays
        // size of this array is not determined yet, should be gotten from scan
        
        // the grid size we process now is total_size;
        int block_size_1d = 512; // can ajust
        int num_block_1d = (total_size_rd+block_size_1d-1)/block_size_1d;
        
        // double time0 = CycleTimer::currentSeconds();
        cudaMalloc((void **)&circ_cover_flag, sizeof(array_type) * total_size_rd);
        cudaMalloc((void **)&circ_cover_id, sizeof(int) * total_size);
        cudaMalloc((void **)&separators, sizeof(int) * num_total_blocks);
        cudaDeviceSynchronize();

        // double time1 = CycleTimer::currentSeconds();
        // printf("step 0 %f s\n",time1-time0);

        //step1: give status  0/1 to the circ_cover_flag based on coverage
        findCircsInBlock<<<num_block_1d,block_size_1d>>> (circ_cover_flag, num_total_blocks, num_blockx, num_blocky, 0, numCircles);
        cudaDeviceSynchronize();

        /*cudaMemcpy(check_flags, circ_cover_flag, total_size_rd * sizeof(int),cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_total_blocks; i++){
        if (i%num_blockx==0) {printf("\n");}
        printf("%d ",  check_flags[i]);
        }
        printf("\n");   
        */
        // // copy the data back to host to print to see our results
        // array_type* checkarray = NULL; 
        // checkarray = (array_type*)malloc(sizeof(array_type) * num_total_blocks);
        // cudaMemcpy(checkarray, circ_cover_flag,  sizeof(array_type) * total_size, cudaMemcpyDeviceToHost);
        
        // for (long i = 0; i < total_size; i++){
        //     printf("check circle %d in block %d : %d\n", i / num_total_blocks, i % num_total_blocks, checkarray[i]);
        // }

        //step2: use a multidimensional scan to find the number of circles each block and this ids
        //save 2 1d arrays: the location increment in the array, the separators.
        //(1) scan the array obtained above
        // double time2 = CycleTimer::currentSeconds();
        // printf("step 1 %f s\n",time2-time1);
        multi_dim_inclusive_scan(total_size_rd, num_circ_rd, num_total_blocks, circ_cover_flag);  //check circ_cover_flag
        //(2) concurrent_write id and separators
        cudaDeviceSynchronize();
        // double time3 = CycleTimer::currentSeconds();
        // printf("step 2(1) %f s\n",time3-time2);
        
        // here we obtain the circ_cover_id
        concurrent_write_ids<<<num_block_1d,block_size_1d>>>(total_size, num_circ_rd, numCircles, num_total_blocks, \
            circ_cover_flag,  circ_cover_id, separators, 0, numCircles); //check circ_cover_id,separators
        cudaDeviceSynchronize();

        // cudaMemcpy(check_sps, separators, num_total_blocks * sizeof(int),cudaMemcpyDeviceToHost);
        // for (int i = 0; i < num_total_blocks; i++){
        //     if (i%num_blockx==0) {printf("\n");} 
        //     printf("%d ",  check_sps[i]);
        // }
        
        // define dim for block
        dim3 blockDimBlock(block_dimx, block_dimy);
        dim3 gridDimBlock(num_blockx, num_blocky);

        // int* separators2; // size:num_total_blocks     [num_circ per block]
        // cudaMalloc((void **)&separators2, sizeof(int) * num_total_blocks);
        // int* check_sps2 = new int[num_total_blocks];
        // findNumCircsInBlock<<<gridDimBlock, blockDimBlock>>> (separators2, num_total_blocks, num_blockx, num_blocky, block_dimx*block_dimy);
        // cudaDeviceSynchronize();
        
        // cudaMemcpy(check_sps2, separators2, num_total_blocks * sizeof(int),cudaMemcpyDeviceToHost);
        // printf("\n");
        // for (int i = 0; i < num_total_blocks; i++){
        //     if (i%num_blockx==0) {printf("\n");} 
        //     printf("%d ",  check_sps2[i]);
        // }

        // double time4 = CycleTimer::currentSeconds();
        // printf("step 2(3) %f s\n",time4-time3);
        //right now, the last 
        //cudaDeviceSynchronize();
        //step3: use the separators and circ_cover_id to render the circle
        

        kernelRenderCircles<<<gridDimBlock, blockDimBlock>>>(separators, circ_cover_id, num_blockx, num_blocky, numCircles);
        cudaDeviceSynchronize();
        // double time5 = CycleTimer::currentSeconds();  
        // printf("step 3 %f s \n",time5-time4); 
    } 
    else if (numCircles < 100000){
        int partitionNum = 1;
        int numCirclesPerPartition = numCircles / partitionNum;
        int num_circ_rd_p = nextPow2(numCirclesPerPartition);

        long total_size_p = numCirclesPerPartition * num_total_blocks;
        long total_size_rd_p = num_circ_rd_p * num_total_blocks;

        // int* separators; // size:num_total_blocks     [num_circ per block]
        // array_type* circ_cover_flag; // the most big array [0,1]
        // int* circ_cover_id;

        // the grid size we process now is total_size;
        int block_size_1d = 512; // can ajust
        int num_block_1d = (total_size_rd_p + block_size_1d-1)/block_size_1d;

        double time0 = CycleTimer::currentSeconds();
        cudaMalloc((void **)&circ_cover_flag, sizeof(array_type) * total_size_rd_p);
        cudaMalloc((void **)&circ_cover_id, sizeof(int) * total_size_p);
        cudaMalloc((void **)&separators, sizeof(int) * num_total_blocks);
        
        int* check_sps = new int[num_total_blocks];

        double time1 = CycleTimer::currentSeconds();
        printf("step 0 %f s\n",time1-time0);
        double time2_sum = 0;
        double time3_sum = 0;
        double time4_sum = 0;
        double time5_sum = 0;
        double timeC_sum = 0;

        for (int i = 0; i < partitionNum; i++){
            double time1_n = CycleTimer::currentSeconds();
            //step1: give status  0/1 to the circ_cover_flag based on coverage
            findCircsInBlock<<<num_block_1d,block_size_1d>>> (circ_cover_flag, num_total_blocks, num_blockx, num_blocky, i, numCirclesPerPartition);
            cudaDeviceSynchronize();
            //step2: use a multidimensional scan to find the number of circles each block and this ids
            //save 2 1d arrays: the location increment in the array, the separators.
            //(1) scan the array obtained above
            double time2 = CycleTimer::currentSeconds();
            time2_sum += (time2 - time1_n);
            multi_dim_inclusive_scan(total_size_rd_p, num_circ_rd_p, num_total_blocks, circ_cover_flag);  //check circ_cover_flag
            //(2) concurrent_write id and separators
            cudaDeviceSynchronize();
            double time3 = CycleTimer::currentSeconds();
            time3_sum += (time3 - time2);

            // here we obtain the circ_cover_id
            concurrent_write_ids<<<num_block_1d,block_size_1d>>>(total_size_p, num_circ_rd_p, numCircles, num_total_blocks, \
                circ_cover_flag,  circ_cover_id, separators, 0, numCircles); //check circ_cover_id,separators
            cudaDeviceSynchronize();
            double time4 = CycleTimer::currentSeconds();
            time4_sum += (time4 - time3);

            cudaMemcpy(check_sps, separators, num_total_blocks * sizeof(int),cudaMemcpyDeviceToHost);
            for (int i = 0; i < num_total_blocks; i++){
                if (i%num_blockx==0) {printf("\n");} 
                printf("%d ",  check_sps[i]);
            }
            
            // define dim for block
            dim3 blockDimBlock(block_dimx, block_dimy);
            dim3 gridDimBlock(num_blockx, num_blocky);
            double timeC = CycleTimer::currentSeconds();
            // time4_sum += (time4 - time3);
            int* separators2; // size:num_total_blocks     [num_circ per block]
            cudaMalloc((void **)&separators2, sizeof(int) * num_total_blocks);
            int* check_sps2 = new int[num_total_blocks];

            findNumCircsInBlock<<<gridDimBlock, blockDimBlock>>> (separators2, num_total_blocks, num_blockx, num_blocky, block_dimx*block_dimy);
            cudaDeviceSynchronize();

            double time4n = CycleTimer::currentSeconds();
            timeC_sum += (time4n - timeC);
            
            cudaMemcpy(check_sps2, separators2, num_total_blocks * sizeof(int),cudaMemcpyDeviceToHost);
            printf("\n");
            for (int i = 0; i < num_total_blocks; i++){
                if (i%num_blockx==0) {printf("\n");} 
                printf("%d ",  check_sps2[i]);
            }

            //step3: use the separators and circ_cover_id to render the circle
            // define dim for block
            // dim3 blockDimBlock(block_dimx, block_dimy);
            // dim3 gridDimBlock(num_blockx, num_blocky);

            kernelRenderCircles<<<gridDimBlock, blockDimBlock>>>(separators, circ_cover_id, num_blockx, num_blocky, numCirclesPerPartition);
            cudaDeviceSynchronize();
            double time5 = CycleTimer::currentSeconds();
            time5_sum += (time5 - time4n);
        }
        printf("step 1 %f s\n",time2_sum);
        printf("step 2(1) %f s\n",time3_sum);
        printf("step 2(3) %f s\n",time4_sum);
        printf("step 2(c) %f s\n",timeC_sum);
        printf("step 3 %f s \n",time5_sum);
    }
    else{
        int partitionNum = 1;
        int numCirclesPerPartition = numCircles / partitionNum;
        int num_circ_rd_p = nextPow2(numCirclesPerPartition);

        long total_size_p = numCirclesPerPartition * num_total_blocks;
        long total_size_rd_p = num_circ_rd_p * num_total_blocks;

        // int* separators; // size:num_total_blocks     [num_circ per block]
        // array_type* circ_cover_flag; // the most big array [0,1]
        // int* circ_cover_id;

        // the grid size we process now is total_size;
        int block_size_1d = 512; // can ajust
        int num_block_1d = (total_size_rd_p + block_size_1d-1)/block_size_1d;

        double time0 = CycleTimer::currentSeconds();
        cudaMalloc((void **)&circ_cover_flag, sizeof(array_type) * total_size_rd_p);
        cudaMalloc((void **)&circ_cover_id, sizeof(int) * total_size_p);
        cudaMalloc((void **)&separators, sizeof(int) * num_total_blocks);
        int* check_sps = new int[num_total_blocks];

        double time1 = CycleTimer::currentSeconds();
        printf("step 0 %f s\n",time1-time0);
        double time2_sum = 0;
        double time3_sum = 0;
        double time4_sum = 0;
        double time5_sum = 0;
        double timeC_sum = 0;

        for (int i = 0; i < partitionNum; i++){
            double time1_n = CycleTimer::currentSeconds();
            //step1: give status  0/1 to the circ_cover_flag based on coverage
            findCircsInBlock<<<num_block_1d,block_size_1d>>> (circ_cover_flag, num_total_blocks, num_blockx, num_blocky, i, numCirclesPerPartition);
            cudaDeviceSynchronize();
            //step2: use a multidimensional scan to find the number of circles each block and this ids
            //save 2 1d arrays: the location increment in the array, the separators.
            //(1) scan the array obtained above
            double time2 = CycleTimer::currentSeconds();
            time2_sum += (time2 - time1_n);

            multi_dim_inclusive_scan(total_size_rd_p, num_circ_rd_p, num_total_blocks, circ_cover_flag);  //check circ_cover_flag
            //(2) concurrent_write id and separators
            cudaDeviceSynchronize();
            double time3 = CycleTimer::currentSeconds();
            time3_sum += (time3 - time2);

            // here we obtain the circ_cover_id
            concurrent_write_ids<<<num_block_1d,block_size_1d>>>(total_size_p, num_circ_rd_p, numCirclesPerPartition, num_total_blocks, \
                circ_cover_flag,  circ_cover_id, separators, i, numCirclesPerPartition); //check circ_cover_id,separators
            cudaDeviceSynchronize();
            double time4 = CycleTimer::currentSeconds();
            time4_sum += (time4 - time3);
            cudaMemcpy(check_sps, separators, num_total_blocks * sizeof(int),cudaMemcpyDeviceToHost);
            for (int i = 0; i < num_total_blocks; i++){
                if (i%num_blockx==0) {printf("\n");} 
                printf("%d ",  check_sps[i]);
            }
            
            // define dim for block
            dim3 blockDimBlock(block_dimx, block_dimy);
            dim3 gridDimBlock(num_blockx, num_blocky);
            double timeC = CycleTimer::currentSeconds();
            // time4_sum += (time4 - time3);
            int* separators2; // size:num_total_blocks     [num_circ per block]
            cudaMalloc((void **)&separators2, sizeof(int) * num_total_blocks);
            int* check_sps2 = new int[num_total_blocks];

            findNumCircsInBlock<<<gridDimBlock, blockDimBlock>>> (separators2, num_total_blocks, num_blockx, num_blocky, block_dimx*block_dimy);
            cudaDeviceSynchronize();
            double time4n = CycleTimer::currentSeconds();
            timeC_sum += (time4n - timeC);

            cudaMemcpy(check_sps2, separators2, num_total_blocks * sizeof(int),cudaMemcpyDeviceToHost);
            printf("\n");
            for (int i = 0; i < num_total_blocks; i++){
                if (i%num_blockx==0) {printf("\n");} 
                printf("%d ",  check_sps2[i]);
            }
            

            kernelRenderCircles<<<gridDimBlock, blockDimBlock>>>(separators, circ_cover_id, num_blockx, num_blocky, numCirclesPerPartition);
            cudaDeviceSynchronize();
            double time5 = CycleTimer::currentSeconds();
            time5_sum += (time5 - time4n);
        }
        printf("step 1 %f s\n",time2_sum);
        printf("step 2(1) %f s\n",time3_sum);
        printf("step 2(3) %f s\n",time4_sum);
        printf("step 2(c) %f s\n",timeC_sum);
        printf("step 3 %f s \n",time5_sum);
    }

    //step4: small size
    //step4: small size

//-------
    
//    cudaFree(circ_cover_flag);
//    cudaFree(circ_cover_id);
//    cudaFree(separators);
    
}
