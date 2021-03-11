#include <getopt.h>
#include <string>
#include "ppm.h"
#include "image.h"
#include "histo.h"

unsigned int *histogramCuda(unsigned char *data, int image_len_in_byte, unsigned int num_bins);
unsigned int *histogramCpu(unsigned char *data, int image_len_in_byte, unsigned int num_bins);

// return GB/s
float toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -i  --image <image file>  Input image in ppm format\n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char** argv) {

    std::string filename("utaustin.ppm");
    unsigned int num_bins = 32;
    // parse commandline options ////////////////////////////////////////////
    int opt;
    extern char *optarg;
    static struct option long_options[] = {
        {"image",  1, 0, 'n'},
        {"bins",  32, 0, 'b'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?i:b:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'i':
            printf("filename:%s\n", optarg);
            filename = optarg;
            break;
        case 'b':
            printf("bin #:%s\n", optarg);
            num_bins = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    ImageChar *image = readPPMImageChar(filename.c_str());

    printCudaInfo();
    
    unsigned int *histo = histogramCuda(image->data, 4*image->getSize(), num_bins);
    unsigned int *histo_ref = histogramCpu(image->data, 4*image->getSize(), num_bins);

    if (histo != NULL && compareHisto(histo, histo_ref, num_bins)) {
        printf("Success\n");
    } else {
        printf("Failed\n");
    }
    if (histo != NULL)
        free(histo);
    free(histo_ref);


    return 0;
}
