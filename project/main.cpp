#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

#include "CycleTimer.h"

void printCudaInfo();



int main(int argc, char** argv)
{
    // All the variables should be float

    // step 1 (input): read or calculate  parameters from "input"
    // and print out information: lxd, nx, ny, Mt

    // step 2 (setup): pass the parameters to constant memory and 
    // allocate and initialize 1_D arrays on CPU/GPU  x: size nx+1, range [0,lxd], y: size ny+1, range [0, lyd]
    // you should get dx = dy = lxd/nx = lyd/ny
    // allocate 1_D arrays on CPU: psi, phi, U of size (nx+3)*(ny+3) -- these are for I/O
    // allocate 1_D arrays on GPU: psi_old/psi_new, phi_old/phi_new, U_old/U_new, same size as before


    // step 3 (time marching): call the kernels Mt times
    
    // step 4: save the psi, phi, U to a .mat file


    return 0;
}
