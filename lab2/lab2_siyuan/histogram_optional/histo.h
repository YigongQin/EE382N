#ifndef __HISTO_H__
#define __HISTO_H__
//struct Histogram { 
//  unsigned int *r_histo;
//  unsigned int *g_histo;
//  unsigned int *b_histo;
//  unsigned long len;
//  Histogram(unsigned long size) { 
//    r_histo = new unsigned int[size];
//    g_histo = new unsigned int[size];
//    b_histo = new unsigned int[size];
//    len = size;
//  } 
//  Histogram(const Histogram &that) {
//    r_histo = that.r_histo;
//    g_histo = that.g_histo;
//    b_histo = that.b_histo;
//    len = that.len;
//  }
//};
#include <stdlib.h>
#include <stdio.h>
bool compareHisto(unsigned int *histoA, unsigned int *histoB, int histo_len);
void printCudaInfo();
enum {
  kR=0,
  kG=1,
  kB=2
};
#endif
