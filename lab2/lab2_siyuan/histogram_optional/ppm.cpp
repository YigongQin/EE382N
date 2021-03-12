#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <cstring>
#include "image.h"
#include "util.h"
using namespace std;


// writePPMImage --
//
// assumes input pixels are float4
// write 3-channel (8 bit --> 24 bits per pixel) ppm
void
writePPMImage(const Image* image, const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename);
        exit(1);
    }

    // write ppm header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", image->width, image->height);
    fprintf(fp, "255\n");

    for (int j=image->height-1; j>=0; j--) {
        for (int i=0; i<image->width; i++) {

            const float* ptr = &image->data[4 * (j*image->width + i)];

            char val[3];
            val[0] = static_cast<char>(255.f * CLAMP(ptr[0], 0.f, 1.f));
            val[1] = static_cast<char>(255.f * CLAMP(ptr[1], 0.f, 1.f));
            val[2] = static_cast<char>(255.f * CLAMP(ptr[2], 0.f, 1.f));

            fputc(val[0], fp);
            fputc(val[1], fp);
            fputc(val[2], fp);
        }
    }

    fclose(fp);
    printf("Wrote image file %s\n", filename);
}

Image *
readPPMImage(const char *filename)
{
  int width = -1;
  int height = -1;
  float max_color = -1;
  Image *image = NULL;
  FILE *infile = fopen(filename,"r");
  if (infile) {
    char header[3];
    // read the magic number, if it is read successfully then continue to get
    // the remaining header information. otherwise display an error
    if (fscanf(infile, "%2s\n", header)==1) {
      cout << header << endl;       
      // skip lines of comments beginning with # get the first character
      char temp = fgetc(infile);
  
      // check to see if the character marks the beginning of a comment
      // if true then continue reading characters until all comments have
      // been read. Otherwise if the character does not mark the beginning
      // of a comment, then put it back
      if (temp == '#')
        while (temp!='\n')
          temp = fgetc(infile);       
      else
        ungetc(temp,infile);
      
      // Read the width, height and range, if it is read successfully then
      // continue to get the data information. Otherwise display an error
      if (fscanf(infile, "%d %d\n%f\n", &width, &height, &max_color)==3) {
        cout << width << " " << height << " " << max_color << "\n";
        if (!strcmp(header, "P3") == 0 && !strcmp(header, "P6") == 0) {
          cerr<<"Wrong magic number (" << header << ") in PPM header." << endl; exit(1);
        }
        cout << "h:" << height << ", w:"<< width << endl;
        image = new Image(width, height);
        // get the data
        for (int j=height-1; j>=0; j--) { // scan line
          for (int i=0; i<width; i++) {   // pixel in a line
             float *ptr = &image->data[4 * (j*width + i)];
             // get the RGB values
             ptr[0] = (float)fgetc(infile) / max_color;  
             ptr[1] = (float)fgetc(infile) / max_color;  
             ptr[2] = (float)fgetc(infile) / max_color;
             ptr[3] = 0.; // kyushick: what is this dimension for?
//             cout << ptr[0] << " " << ptr[1] << " " << ptr[2] << endl;
          }  
        }
      }
      else {
        cerr<<"Problem reading PPM header file - width,height,max color."<<endl; exit(1);
      }
    }
    else {
      cerr<< "Problem reading PPM header file - magic number."<<endl; exit(1);
    }
  } else {
     cerr << "The file, " << filename << ", cannot be opened.";
  }
  fclose(infile);
  return image;
}

ImageChar *
readPPMImageChar(const char *filename)
{
  int width = -1;
  int height = -1;
  int max_color = -1;
  ImageChar *image = NULL;
  FILE *infile = fopen(filename,"r");
  if (infile) {
    char header[3];
    // read the magic number, if it is read successfully then continue to get
    // the remaining header information. otherwise display an error
    if (fscanf(infile, "%2s\n", header)==1) {
      cout << header << endl;       
      // skip lines of comments beginning with # get the first character
      char temp = fgetc(infile);
  
      // check to see if the character marks the beginning of a comment
      // if true then continue reading characters until all comments have
      // been read. Otherwise if the character does not mark the beginning
      // of a comment, then put it back
      if (temp == '#')
        while (temp!='\n')
          temp = fgetc(infile);       
      else
        ungetc(temp,infile);
      
      // Read the width, height and range, if it is read successfully then
      // continue to get the data information. Otherwise display an error
      if (fscanf(infile, "%d %d\n%d\n", &width, &height, &max_color)==3) {
        cout << width << " " << height << " " << max_color << "\n";
        if (!strcmp(header, "P3") == 0 && !strcmp(header, "P6") == 0) {
          cerr<<"Wrong magic number (" << header << ") in PPM header." << endl; exit(1);
        }
        image = new ImageChar(width, height);
        // get the data
        for (int j=height-1; j>=0; j--) { // scan line
          for (int i=0; i<width; i++) {   // pixel in a line
             unsigned char *ptr = &image->data[4 * (j*width + i)];
             // get the RGB values
             const unsigned char r = fgetc(infile);  
             const unsigned char g = fgetc(infile);  
             const unsigned char b = fgetc(infile);
             ptr[0] = (r > max_color)? (char)max_color : r;  
             ptr[1] = (g > max_color)? (char)max_color : g; 
             ptr[2] = (b > max_color)? (char)max_color : b;
             ptr[3] = 0.; // kyushick: what is this dimension for?
            // printf("r:%u g:%u b:%u\n", ptr[0], ptr[1], ptr[2]);
          }  
        }
      }
      else {
        cerr<<"Problem reading PPM header file - width,height,max color."<<endl; exit(1);
      }
    }
    else {
      cerr<< "Problem reading PPM header file - magic number."<<endl; exit(1);
    }
  } else {
     cerr << "The file (" << filename << ") cannot be opened." <<endl; exit(1);
  }
  fclose(infile);
  return image;
}

void printHisto(unsigned int *histo, unsigned int len) {
  for (unsigned int i=0; i<len; i++) 
    cout << i << ":" << histo[i] << endl;
}
