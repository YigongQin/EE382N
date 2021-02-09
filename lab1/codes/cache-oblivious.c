#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include<math.h>
#include<string.h>

// cache-oblivious algorithm
// args: A, B, C, size of submatrix, id for A, B and C 
/*void cache_oblv(float **A, float **B, float **C, int N, int depth, int* Ai, int* Bi, int* Ci, int* Aj, int* Bj, int* Cj) {
  float sum;
  int   i;
  int   j;
  int   k;
  if (N==1) {
     C[0][0]=A[0][0]*B[0][0];
  }else{
   // call cache-oblivious eight times add corresponding values to C
   cache_oblv

 
  }


}*/

void cache_oblv(float **A, float **B, float **C, int N, int Ai, int Aj, int Bi, int Bj, int Ci, int Cj) {
 
  int bs = N/2.0;
 
  if (N==1) {
     C[Ci][Cj]=A[Ai][Aj]*B[Bi][Bj];
  }else{

    cache_oblv(A,B,C,bs, Ai, Aj, Bi, Bj, Ci, Cj); cache_oblv(A,B,C,bs, Ai, Aj+bs, Bi+bs, Bj, Ci, Cj);
    cache_oblv(A,B,C,bs, Ai, Aj, Bi, Bj+bs, Ci, Cj+bs); cache_oblv(A,B,C,bs, Ai, Aj+bs, Bi+bs, Bj+bs, Ci, Cj+bs);
    cache_oblv(A,B,C,bs, Ai+bs, Aj, Bi, Bj, Ci+bs, Cj); cache_oblv(A,B,C,bs, Ai+bs, Aj+bs, Bi+bs, Bj, Ci+bs, Cj);
    cache_oblv(A,B,C,bs, Ai+bs, Aj, Bi, Bj+bs, Ci+bs, Cj+bs); cache_oblv(A,B,C,bs, Ai+bs, Aj+bs, Bi+bs, Bj+bs, Ci+bs, Cj+bs);


  }

}

// calculate C = AxB
void matmul(float **A, float **B, float **C, int Nmat) {
  float sum;
  int   i;
  int   j;
  int   k;
  int M, N, P;
  M=Nmat; N=Nmat; P=Nmat;
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

int main(int argc, char **argv) {
  int Nmat = atoi(argv[1]);
  float** A;
  float** B;
  float** C;
  int M, N, P;
  int depth= log2f(Nmat);
  int Ai[depth],Bi[depth],Ci[depth],Aj[depth],Bj[depth],Cj[depth];
  memset(Ai, 0, depth*sizeof(int));memset(Bi, 0, depth*sizeof(int));memset(Ci, 0, depth*sizeof(int));
  memset(Aj, 0, depth*sizeof(int));memset(Bj, 0, depth*sizeof(int));memset(Aj, 0, depth*sizeof(int));  
  printf("depth %d\n", depth);printf("M=%d\n",M);
  for (int i=0;i<depth;i++ ){printf("%d ",Ai[i]);}
  M = Nmat; N = Nmat; P = Nmat;
  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);
  
  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  cache_oblv(A, B, C, Nmat, 0, 0, 0, 0, 0, 0);

  return (0);
}



