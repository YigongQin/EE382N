/*
 * @@Description: This file is used to implement cache aware matrix multiplication
 * @@Author: Hanqing Zhu
 * @Date: 2021-02-08 10:34:14
 * @LastEditors: Hanqing Zhu
 * @LastEditTime: 2021-02-09 21:45:12
 * @FilePath: \Sp21CALab\cache_aware.c
 */

#include <stdlib.h>
#include <stdio.h>
// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 4096
#define N 4096
#define P 4096

// L1 cache size 32KB   BSIZE1 <= 32 = 52
// L2 cache size 1MB    BSIZE2 <= 256 = 295
// L2+L3 57MB           BSIZE3 <= 1024 = 2231
// #define BSIZE1 32
// #define BSIZE2 256
// #define BSIZE3 1024

int BSIZE1;
int BSIZE2;
// int BSIZE3;

// calculate C = AxB
void matmul(float **A, float **B, float **C) {
  float sum;
  int   i;
  int   j;
  int   k;
 
  for (i=0; i<M; i++) {
    // for each row of C
    for (j=0; j<N; j++) {
      // for each column of C
      sum = 0.0f; // temporary value
      for (k=0; k<P; k++) {
        // dot product of row from A and column from B
        sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

// function to recursive mul with block division
void matmul_aware_recur(float **A, float **B, float **C, int i0, int i1, int j0, int j1, int k0, int k1, int level)
{
    // printf("level:%d\n", level);
    float sum;
    int i;
    int j;
    int k;

    if(level == 0){
      for(i = i0; i < i1; i++)
      {
        for(j = j0; j < j1; j++)
        {
          sum = C[i][j];
          for(k = k0; k < k1; k++)
          {
            sum += A[i][k] * B[k][j];
          }
          C[i][j] = sum;
        }
      }
    }
    else
    {
      int bsize;
      // check locate to which memory hierarchy
      // if case 3: M > max cache size such that we need to split the matrix to make sure L3 can hold
      switch (level)
      {
      case 1:
        bsize = BSIZE1;
        break;
      case 2:
        bsize = BSIZE2;
        break;
      default:
        break;
      }
      for (i = i0; i < i1; i+=bsize)
      {
        for (j = j0; j < j1; j+=bsize)
        {
          for (k = k0; k < k1; k+=bsize)
          {
            matmul_aware_recur(A, B, C, i, i+bsize, j, j+bsize, k, k+bsize, level-1);
          }
        }
      }
    }
}

// function to compute cache aware mat mul
void matmul_aware(float **A, float **B, float **C)
{
    // compute locality level
    int level;
    if(M <= BSIZE1)
    {
        level = 0;
    }
    else if(M <= BSIZE2)
    {
        level = 1;
    }
    else
    {
        level = 2;
    }
    // printf("level:%d\n", level);

    matmul_aware_recur(A, B, C, 0, M, 0, N, 0, P, level);
}


// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float*** A, int m, int n) {
  float **T = 0;
  int i;
 
  T = (float**)malloc( m*sizeof(float*));
  for ( i=0; i<m; i++ ) {
     T[i] = (float*)malloc(n*sizeof(float));
  }
  *A = T;
}
 
int main(int argc, char** argv) {
  float** A;
  float** B;
  float** C;
  
  // obtain the block size for different memory
  BSIZE1 = atoi(argv[1]);
  BSIZE2 = atoi(argv[2]);
  // BSIZE3 = atoi(argv[3]);

  // printf("BSIZE1 = %d, BSIZE2 = %d, BSIZE3 = %d", BSIZE1, BSIZE2, BSIZE3);
  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);
   // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.
  // matmul(A, B, C);

  matmul_aware(A, B, C);
 
  return (0);
}

