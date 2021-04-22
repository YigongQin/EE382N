#ifndef __PPM_H__
#define __PPM_H__

struct Image;
struct ImageChar;

void writePPMImage(const Image* image, const char *filename);
ImageChar *readPPMImageChar(const char *filename);
Image *readPPMImage(const char *filename);
void printHisto(unsigned int *histo, unsigned int len);

#endif
