#ifndef __SAXPY_H__
#define __SAXPY_H__
float toBW(long bytes, float sec);
void saxpyCuda(long N, float alpha, float* x, float* y, float* result, int partitions);
void printCudaInfo();
void getArrays(int size, float **xarray, float **yarray, float **resultarray);
void freeArrays(float *xarray, float *yarray, float *resultarray);
bool check_saxpy(long N, float* a, float* b);
extern double timeCopyH2DAvg;
extern double timeCopyD2HAvg;
extern double timeKernelAvg;
extern double totalTimeAvg;
#endif
