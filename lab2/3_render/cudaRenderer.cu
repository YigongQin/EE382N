#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "device_launch_parameters.h"

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

// add thrust
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/execution_policy.h>

#define THREADS_PER_BLOCK 256

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

////////////////////////////////////////////////////////////////////////////////////////
// Add our impl
///////////////////////////////////////////////////////////////////////////////////////


// scan codes
// Rounding up to next power of 2
static inline int nextPow2(int n)
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

__global__ void
upsweep(int N, int twod, int twod1, int* output) {

    // upsweep phase of exclusive scan

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (twod==N){
          if (index==N-1){output[index]=0;}}
    else{
      if (index < N && (index % twod1 ==0) )
       output[index+ twod1 -1] += output[index+twod -1];}
}

__global__ void
downsweep(int N, int twod, int twod1, int* output) {

    // downsweep phase of exclusive scan
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N && (index % twod1 ==0) ){
       int t  = output[index+twod -1];
       output[index+ twod -1] = output[index+ twod1 -1];;
       output[index+ twod1 -1] += t;}
}

__global__ void debug_kernel(int N, int* out_array){
    for(int i = 0; i < N; i++){
   printf("out_array[%d] = %d\n",i, out_array[i]);
    }
    printf("\n");
  }

cudaError_t cudaCheckError(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

struct Pair {
	int circle;
	int cell;
	bool operator < (const Pair &p) const {
		//return (cell < p.cell || (cell == p.cell && circle < p.circle));
        return (cell < p.cell);//maybe this is enough?
	}
	Pair(){}
};

__global__ void getPairNums(int* pairNums, int cellWidth, int cellHeight){
    // copy from kernelRenderCircles
    int circleIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (circleIdx >= cuConstRendererParams.numCircles) {
        return;
    }

    int index3 = 3 * circleIdx;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[circleIdx];

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

    // compute the start region and end region occupied by circle
    int startRegionX = screenMinX / cellWidth;
    int endRegionX   = (screenMaxX + cellWidth - 1) / cellWidth;
    int startRegionY = screenMinY / cellHeight;
    int endRegionY   = (screenMaxY + cellHeight - 1) / cellHeight;

    pairNums[circleIdx] = (endRegionY - startRegionY) * (endRegionX - startRegionX);
}

__global__ void makePairs(Pair* pairs, int* pairNums, int cellWidth, int cellHeight, int cellNumX, int cellNumY){
    // copy from kernelRenderCircles
    int circleIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (circleIdx >= cuConstRendererParams.numCircles) {
        return;
    }

    int index3 = 3 * circleIdx;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[circleIdx];

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

    // compute the start region and end region occupied by circle
    int startRegionX = screenMinX / cellWidth;
    int endRegionX   = (screenMaxX + cellWidth - 1) / cellWidth;
    int startRegionY = screenMinY / cellHeight;
    int endRegionY   = (screenMaxY + cellHeight - 1) / cellHeight;

    // Pair the region with corresponding circle
    int regionIdx; // combine with the previous cells equal to real regionIdx by circle order

    for (int xIdx = startRegionX; xIdx < endRegionX; xIdx++){
        for (int yIdx = startRegionY; yIdx < endRegionY; yIdx++){
            // regionIdx = (xIdx - startRegionX) * (endRegionY - startRegionY) + (yIdx - startRegionY);
            regionIdx = (yIdx - startRegionY) * (endRegionX - startRegionX) + xIdx - startRegionX;
            // update the pairs which bind current circle with it occupied regions
            pairs[pairNums[circleIdx] + regionIdx].circle = circleIdx;
            pairs[pairNums[circleIdx] + regionIdx].cell = xIdx + yIdx * cellNumX;
        }
    }

}

__global__ void getBounds(Pair* pairs, int* start, int* end, int pairsLength){
    __shared__ Pair cache[THREADS_PER_BLOCK + 1];
    int threadId = threadIdx.x;
    // obatin the place of strat and end of cell in pairs
    int pairId = blockDim.x * blockIdx.x + threadIdx.x;

    if (pairId >= pairsLength){
        return;
    }

    cache[threadId] = pairs[pairId];
	if (threadId == THREADS_PER_BLOCK - 1 && pairId < pairsLength - 1) cache[THREADS_PER_BLOCK] = pairs[pairId + 1];
	__syncthreads();


	if (pairId == pairsLength - 1) {
		end[cache[threadId].cell] = pairsLength;
		return;
	}
	if (pairId == 0) {
		start[cache[threadId].cell] = 0;
	}
	if (cache[threadId].cell != cache[threadId + 1].cell) {
		end[cache[threadId].cell] = pairId + 1;
		start[cache[threadId + 1].cell] = pairId + 1;
	}
	return;

    // update start and end

}

////////////////////////////////////////////////////////////////////////////////////////

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
__global__ void kernelRenderCircles_simple() {

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

__global__ void kernelRenderCircles(Pair* pairs, int* start, int* end, int cellNumX, int cellNumY){
    
    int regionIdx = blockIdx.y * cellNumX + blockIdx.x;

    // obtain the start and end of circle idx for current region 
    int startCircleIdx = start[regionIdx];
    int endCircleIdx   = end[regionIdx];

    if (startCircleIdx == -1)return;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the region
    // update each pixel based on given sequence of circles on each region
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelY >= imageHeight || pixelX >= imageWidth) return;

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
        invHeight * (static_cast<float>(pixelY) + 0.5f));
    
    float4 *imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);

    // iterate over all circles on this region
    for (int idx = startCircleIdx; idx < endCircleIdx; idx ++){
        // update pixel under circle order
        int circleIdx = pairs[idx].circle;
        int index3 = 3 * circleIdx;
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        // float  rad = cuConstRendererParams.radius[circleIdx];
        shadePixel(circleIdx, pixelCenterNorm, p, imgPtr);
    }
    // write back
    // (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]) = imgPtr;
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

void exclusive_scan( int length, int* device_result)
{

    // length here have been rounded to power of two, device_result is device array to be scanned. 
    int blocksize = 256;
    int num_blocks = (length+blocksize-1)/blocksize;
    printf("block size = %d, number of blocks %d \n",blocksize,num_blocks);
    for (int twod =1; twod <=length; twod *=2){
        int twod1 = twod*2;
            upsweep<<< num_blocks, blocksize  >>>(length, twod, twod1, device_result);
    }
    //device_result[length-1]=0;
    for (int twod = length/2; twod >=1; twod /=2){
        int twod1 = twod*2;
            downsweep<<< num_blocks, blocksize  >>>(length, twod, twod1, device_result);
    }
}


void
CudaRenderer::render() {

    // 256 threads per block is a healthy number
    dim3 blockDim(THREADS_PER_BLOCK, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    int cellWidth=32;
    int cellHeight=32;
    int cellNumX=(image->width + cellWidth - 1)/cellWidth;
    int cellNumY=(image->height + cellHeight -1)/cellHeight;

    //pairNums: each entry contains the number of cells interfering with a circle.
    int* pairNums;
    int pairNumsLength = nextPow2(numCircles);

    // check whether it is equal which will make exclusive scan fail
    if (numCircles == pairNumsLength){
        pairNumsLength = nextPow2(numCircles + 1);
    }

    cudaCheckError(cudaMalloc((void**)(&pairNums), pairNumsLength*sizeof(int)));
    cudaMemset(pairNums, 0, sizeof(int) * pairNumsLength);

    //calculate pairNums
    getPairNums<<<gridDim, blockDim>>>(pairNums, cellWidth, cellHeight);
	cudaCheckError(cudaDeviceSynchronize());
    
    //accumulate pairNums to itself with csr format
    exclusive_scan(pairNumsLength, pairNums);
    cudaCheckError(cudaDeviceSynchronize()); // we need add this synchronization
   
    //pairs: each entry maps 1 circle to 1 cell
    Pair* pairs;
    int pairsLength;
    // The overall pairNums should be precisely equal to the pairNums[numCircles] where we set a nextPow2 value with extra values to be 0 
    cudaMemcpy(&pairsLength, (void*)(&pairNums[numCircles]), sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMalloc((void**)(&pairs), sizeof(Pair) * pairsLength));
    
    //calculate pairs
    makePairs<<<gridDim, blockDim>>>(pairs, pairNums, cellWidth, cellHeight, cellNumX, cellNumY);
	cudaCheckError(cudaDeviceSynchronize());

    //sort pairs by cell (done in host)
    Pair* host_pairs;
	host_pairs = new Pair[pairsLength];
	cudaMemcpy(host_pairs, pairs, sizeof(Pair) * pairsLength, cudaMemcpyDeviceToHost);
	thrust::sort(thrust::host, host_pairs, host_pairs + pairsLength);
	cudaCheckError(cudaDeviceSynchronize());	
	cudaMemcpy(pairs, host_pairs, sizeof(Pair) * pairsLength, cudaMemcpyHostToDevice);

    //boundaries in pairs for each cell
    int *start, *end;
	cudaCheckError(cudaMalloc((void**)(&start), sizeof(int) * cellNumX * cellNumY));
	cudaMemset(start, -1, sizeof(int) * cellNumX * cellNumY);
	cudaCheckError(cudaMalloc((void**)(&end), sizeof(int) * cellNumX * cellNumY));

    //calculate boundaries
    getBounds<<<((pairsLength-1)+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(pairs, start, end, pairsLength);
	cudaCheckError(cudaDeviceSynchronize());

    //parallel render each cell
    dim3 blockDimCell(cellWidth, cellHeight);
	dim3 gridDimCell(cellNumX, cellNumY);
    kernelRenderCircles<<<gridDimCell, blockDimCell>>>(pairs, start, end, cellNumX, cellNumY);
    // kernelRenderCircles_simple<<<gridDim, blockDim>>>();
	cudaDeviceSynchronize();

    //cudaFree(totalNum);
    cudaFree(pairNums);
    cudaFree(pairs);
    cudaFree(start);
    cudaFree(end);
}
