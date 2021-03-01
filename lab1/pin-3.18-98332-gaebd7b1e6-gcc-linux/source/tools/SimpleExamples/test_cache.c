#include <stdlib.h>
#include <stdio.h>

const int skip = 4;
const int n = 1*65536;

void f(int* A) {
  int i;
  int tot;

  for (i=0; i<n; i++) {
    A[i*skip] = i;
    tot += A[i*skip];
  }
  printf("%d\n", tot);
} 


int main() {
  int* A = malloc(sizeof(int)*n*skip);
  if (A == 0) {
    exit (-1);
  }

  f(A);
  f(A);

  return (0);
}
