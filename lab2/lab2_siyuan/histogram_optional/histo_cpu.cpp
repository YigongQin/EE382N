#include "histo.h"
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include "common.h"
unsigned int *histogramCpu(unsigned char *data, int image_len_in_byte, unsigned int num_bins)
{
    struct timeval t0, t1;
    gettimeofday(&t0, 0);
    int histo_len_in_byte = 3*num_bins*sizeof(unsigned int); // r,g,b
    unsigned int *histo = (unsigned int *)malloc(histo_len_in_byte);
    memset(histo, 0, histo_len_in_byte);
    unsigned int *r = histo;
    unsigned int *g = histo + num_bins;
    unsigned int *b = histo + num_bins*2;
    int bin_width = (int)ceil(256. / num_bins);
    unsigned int pos = 0;
    for (int i=0; i<image_len_in_byte; i+=4) {
        pos = int(data[i + kR])/bin_width;
        if (pos < num_bins) // check range
            r[pos]++;
        else 
          MYDEBUG("error R %d\n", pos);
        pos = int(data[i + kG])/bin_width;
        if (pos < num_bins) // check range
            g[pos]++;
        else 
          MYDEBUG("error G %d\n", pos);
        pos = int(data[i + kB])/bin_width;
        if (pos < num_bins) // check range
           b[pos]++;
        else 
          MYDEBUG("error B %d\n", pos);
    }
    gettimeofday(&t1, 0);
    long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
    printf("CPU Time  : %9.3lf ms\n", 0.001 * elapsed);
    MYDEBUG("Bin width:%d\n", bin_width);
    return histo;
}

bool compareHisto(unsigned int *histoA, unsigned int *histoB, int num_bins) {
    int histo_len = 3*num_bins;
    bool different = false;
    for (int i=0; i<histo_len; i++) {
        if (histoA[i] != histoB[i]) {
            MYDEBUG("%d %u != %u\n", i, histoA[i], histoB[i]);
            different |= true;
        }
    }
    return (different == false);
}
