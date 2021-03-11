#ifndef  __IMAGE_H__
#define  __IMAGE_H__


struct Image {

    Image(int w, int h) {
        width = w;
        height = h;
        data = new float[4 * width * height];
    }

    void clear(float r, float g, float b, float a) {

        int numPixels = width * height;
        float* ptr = data;
        for (int i=0; i<numPixels; i++) {
            ptr[0] = r;
            ptr[1] = g;
            ptr[2] = b;
            ptr[3] = a;
            ptr += 4;
        }
    }

    int width;
    int height;
    float* data;
    unsigned long getSize() { return ((unsigned long)width * (unsigned long)height); }
//    void print() {
//        unsigned long size = getSize();
//        float* ptr = data;
//        for (int i=0; i<numPixels; i++) {
//            printf("%d, %d, %d\n", (int)ptr[0], (int)ptr[1], (int)ptr[2]);
//            ptr += 4;
//        }
//    }

};

struct ImageChar {

    ImageChar(int w, int h) {
        width = w;
        height = h;
        data = new unsigned char[4 * width * height];
    }

    void clear(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {

        int numPixels = width * height;
        unsigned char* ptr = data;
        for (int i=0; i<numPixels; i++) {
            ptr[0] = r;
            ptr[1] = g;
            ptr[2] = b;
            ptr[3] = a;
            ptr += 4;
        }
    }

    int width;
    int height;
    unsigned char* data;
    unsigned long getSize() { return ((unsigned long)width * (unsigned long)height); }

};


#endif
