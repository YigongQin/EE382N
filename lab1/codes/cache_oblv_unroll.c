#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#define Ncut 8
// cache-oblivious algorithm
void cache_oblv(float **A, float **B, float **C, int N, int Ai, int Aj, int Bi, int Bj, int Ci, int Cj) {
 
  int bs = N/2.0;
 //  printf("the block size: %d\n",bs); 
  if (N==Ncut) {
     float sum;
     // call ijk algorithm
     for (int i=0; i<N; i++) {
        // for each row of C
        for (int j=0; j<N; j++) {
        // for each column of C
            sum = 0.0f; // temporary value
           // for (int k=0; k<N; k++) {
        // dot product of row from A and column from B
            sum += A[Ai+i][Aj+0]*B[Bi+0][Bj+j];sum += A[Ai+i][Aj+1]*B[Bi+1][Bj+j];
            sum += A[Ai+i][Aj+2]*B[Bi+2][Bj+j];sum += A[Ai+i][Aj+3]*B[Bi+3][Bj+j];
            sum += A[Ai+i][Aj+4]*B[Bi+4][Bj+j];sum += A[Ai+i][Aj+5]*B[Bi+5][Bj+j];
            sum += A[Ai+i][Aj+6]*B[Bi+6][Bj+j];sum += A[Ai+i][Aj+7]*B[Bi+7][Bj+j];
          //}
            C[Ci+i][Cj+j] = sum;
    }}
  }else if (N==1){
     C[Ci][Cj]+=A[Ai][Aj]*B[Bi][Bj];
     //printf("%f\n",C[Ci][Cj]);
  }else{

    cache_oblv(A,B,C,bs, Ai, Aj, Bi, Bj, Ci, Cj); cache_oblv(A,B,C,bs, Ai, Aj+bs, Bi+bs, Bj, Ci, Cj);//print_matrix(C,N);
    cache_oblv(A,B,C,bs, Ai, Aj, Bi, Bj+bs, Ci, Cj+bs); cache_oblv(A,B,C,bs, Ai, Aj+bs, Bi+bs, Bj+bs, Ci, Cj+bs);
    cache_oblv(A,B,C,bs, Ai+bs, Aj, Bi, Bj, Ci+bs, Cj); cache_oblv(A,B,C,bs, Ai+bs, Aj+bs, Bi+bs, Bj, Ci+bs, Cj);
    cache_oblv(A,B,C,bs, Ai+bs, Aj, Bi, Bj+bs, Ci+bs, Cj+bs); cache_oblv(A,B,C,bs, Ai+bs, Aj+bs, Bi+bs, Bj+bs, Ci+bs, Cj+bs);

  }

}

void print_matrix(float **A, int N){

      for( int i = 0; i < N; i++){
        for( int j = 0; j < N; j++)
                   { printf(" %f ", A[i][j]);}
                printf( "\n");}
      printf("\n");
}
void assign_values(float **A, int N){

      for( int i = 0; i < N; i++){
        for( int j = 0; j < N; j++)
            {  A[i][j]= (float) rand()/RAND_MAX;}}

}




void transpose(float **A, int N) {
    float temp;
    for(int i = 0; i<N; i++) {
        for (int j = i+1; j<N; j++)  
           {temp = A[j][i]; A[j][i]=A[i][j]; A[i][j] = temp;}
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
  M = Nmat; N = Nmat; P = Nmat;
  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);
  assign_values(A,Nmat);assign_values(B,Nmat); 
  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.
 // print_matrix(A,Nmat);print_matrix(B,Nmat);
 // cache_oblv(A, B, C, Nmat, 0, 0, 0, 0, 0, 0);
 // transpose(B, Nmat);
 // print_matrix(B,Nmat);
  cache_oblv(A, B, C, Nmat, 0, 0, 0, 0, 0, 0);
  //print_matrix(C,Nmat);
  return (0);
}


