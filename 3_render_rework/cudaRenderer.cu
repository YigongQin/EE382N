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

#define array_type short
// #define array_type int
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

    if ( (0< index) && (index < N) && ( (index/dim) % twod1 ==0) ){
         output[index+ dim*(twod-1)] += output[index- dim*1];}
}

//--- the above should be correct

__global__ void
concurrent_write_ids(int total_size, int num_circ, int num_boxes, array_type* circ_cover_flag, int* circ_cover_id, int* separators){

     //
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     int circleid = index/num_boxes;
     int blockid = index-num_boxes*circleid;
     if (index<total_size){
         if (circleid==0){
               if (circ_cover_flag[index]==0){
                   int new_loc = num_circ*blockid;
                   circ_cover_id[new_loc]=0;}}
         else{
               if ( circ_cover_flag[index] - circ_cover_flag[index-num_boxes] ==1){
                int new_loc = blockid*num_circ +circ_cover_flag[index] -1;
                circ_cover_id[new_loc] = circleid; }
              }
         
     }
     //update the separators by the way
     if (index<num_boxes) {separators[index]=circ_cover_flag[(num_circ-1)*num_boxes+index];}
}


void multi_dim_inclusive_scan(int N, int lens, int dim, array_type* device_result){

    int blocksize = 512;
    int num_blocks = (N+blocksize-1)/blocksize;
    printf("N=%d,block size = %d, number of blocks %d \n",N,blocksize,num_blocks); 
    for (int twod =1; twod <lens; twod *=2){
        int twod1 = twod*2;
            incl_sweep_up<<< num_blocks, blocksize  >>>(N, dim, twod, twod1, device_result);        
    }

    for (int twod = lens/4; twod >=1; twod /=2){
        int twod1 = twod*2;
            incl_sweep_down<<< num_blocks, blocksize  >>>(N, dim, twod, twod1, device_result);
    }


}


#include "circleBoxTest.cu_inl"
__global__ void findCircsInBlock(array_type* circ_cover_flag, int num_total_blocks, int num_blockx, int num_blocky) {
    // step1: find the circle idx and find the block idx
    int Idx = blockDim.x * blockIdx.x + threadIdx.x; // B*numCircles

    int circleId = Idx / num_total_blocks; //obtain the circle Id
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
    
    int blockId_dimX = blockId / num_blockx;
    int blockId_dimY = blockId % num_blockx;

    short blockMinX = BLOCK_DIM_X * blockId_dimX;
    short blockMaxX = BLOCK_DIM_X * (blockId_dimX + 1) - 1;
    short blockMinY = BLOCK_DIM_Y * blockId_dimY;
    short blockMaxY = BLOCK_DIM_Y * (blockId_dimY + 1) - 1;  

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
        int index3 = circleIdx * 3;
        // read postion and radius then use shadePixel to update
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        shadePixel(circleIdx, pixelCenterNorm, p, imgPtr);
    }
}

void
CudaRenderer::render() {

    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // int* debug_flag = new int[20];
    //  int debug[20]  = {1,1,0,0,1, 0,0,1,0,0, 0,1,1,0,0, 0,0,0,1,1};
    //  memmove(debug_flag, debug, 20*sizeof(int));
    //  int* debug_flag_result = new int[20];
    //  int* debug_id_result = new int[20];
    //  for (int i = 0; i < 20; i++){
    //      printf("%d\n", debug_flag[i]);
    //  }
    //  int* device_flag;
    //  int* device_id;
    //  int* separat_debug;
    //  int N_rd = nextPow2(4);
    //  int B = 5;
    //  int debug_separators[B];
    //  int total = N_rd*B;
    //  printf("total %d \n",total);
    //  cudaMalloc((void **)&device_flag, sizeof(int) * total);
    //  cudaMalloc((void **)&device_id, sizeof(int) * total);
    //  cudaMalloc((void **)&separat_debug, sizeof(int) * B);

    //  cudaMemcpy(device_flag, debug_flag, total * sizeof(int),cudaMemcpyHostToDevice);
    //  cudaMemset(device_flag, 1, total*sizeof(int));
    //  //multi_dim_inclusive_scan(total, N_rd, B, device_flag);
    //  //concurrent_write_ids<<<10,10>>>(total, N_rd, N, B, device_flag,  device_id, separat_debug);

    //  cudaMemcpy(debug_flag_result, device_flag, total * sizeof(int),cudaMemcpyDeviceToHost);
    //  //cudaMemcpy(debug_id_result, device_id, total * sizeof(int),cudaMemcpyDeviceToHost);
    //  //print
    //  for (int i = 0; i < 20; i++){
    //      printf("%d\n", debug_flag_result[i]);
    //  }
    //  printf("\n");
     
// //------Yigong adding----
    // short, int, long long int should be justified
    int num_circ_rd = nextPow2(numCircles); //rounded numCircles
    array_type* circ_cover_flag; // the most big array [0,1]
    
    int* circ_cover_id; // the most big array
    int block_dimx = 32;
    int block_dimy = 32;
    int num_blockx = (image->width+block_dimx-1)/block_dimx;
    int num_blocky = (image->height+block_dimy-1)/block_dimy;    
    int num_total_blocks = num_blockx*num_blocky;
    long total_size = numCircles*num_total_blocks;
    // long total_size = numCircles;
    //int* circ_loca_ids; // concatenation of num_total_blocks variable-size arrays
    // size of this array is not determined yet, should be gotten from scan
    int* separators; // size:num_total_blocks     [num_circ per block]
    
    // the grid size we process now is total_size;
    int block_size_1d = 512; // can ajust
    int num_block_1d = (total_size+block_size_1d-1)/block_size_1d;
    
    cudaMalloc((void **)&circ_cover_flag, sizeof(array_type) * total_size);
    cudaMemset(circ_cover_flag, 0, sizeof(array_type) * total_size); // set to 0 to prevent error

    cudaMalloc((void **)&circ_cover_id, sizeof(int) * total_size);
    cudaMalloc((void **)&separators, sizeof(int) * num_total_blocks);
    
    cudaDeviceSynchronize();
    //step1: give status  0/1 to the circ_cover_flag based on coverage
    findCircsInBlock<<<num_block_1d,block_size_1d>>> (circ_cover_flag, num_total_blocks, num_blockx, num_blocky);
    cudaDeviceSynchronize();
    
    // copy the data back to host to print to see our results
    // array_type* checkarray = NULL; 
    // checkarray = (array_type*)malloc(sizeof(array_type) * num_total_blocks);
    // cudaMemcpy(checkarray, circ_cover_flag,  sizeof(array_type) * total_size, cudaMemcpyDeviceToHost);
    
    // for (long i = 0; i < total_size; i++){
    //     printf("check circle %d in block %d : %d\n", i / num_total_blocks, i % num_total_blocks, checkarray[i]);
    // }

    //step2: use a multidimensional scan to find the number of circles each block and this ids
    //save 2 1d arrays: the location increment in the array, the separators.
    //(1) scan the array obtained above
    multi_dim_inclusive_scan(total_size, num_circ_rd, num_total_blocks, circ_cover_flag);  //check circ_cover_flag
    //(2) concurrent_write id and separators
    concurrent_write_ids<<<num_block_1d,block_size_1d>>>(total_size, num_circ_rd, num_total_blocks, \
    circ_cover_flag,  circ_cover_id, separators); //check circ_cover_id,separators
    //right now, the last 
    cudaDeviceSynchronize();
    
    //step3: use the separators and circ_cover_id to render the circle
    // define dim for block
    dim3 blockDimBlock(block_dimx, block_dimy);
    dim3 gridDimBlock(num_blockx, num_blocky);

    kernelRenderCircles<<<gridDimBlock, blockDimBlock>>>(separators, circ_cover_id, num_blockx, num_blocky, numCircles);
    cudaDeviceSynchronize();
    
    //step4: small size

//-------
    
    cudaFree(circ_cover_flag);
    cudaFree(circ_cover_id);
    cudaFree(separators);
    
}
