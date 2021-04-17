#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <map>
#include<functional>

#include "CycleTimer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
// #include <mat.h> 

#include <cuda.h> // add cuda support
#include <cuda_runtime.h>
#include <driver_functions.h>

void setup(float param_nx, float param_ny, float param_G, float param_R, float param_delta,
    float param_k, float param_c_infm, float param_Dl, float param_d0, float param_W0, 
    float param_lT, float param_lamd, float param_tau0, float param_c_infty, float param_R_tilde,
    float param_Dl_tilde, float param_lT_tilde, float param_eps, float param_alpha0, float param_dx, 
    float param_dt, float param_asp_ratio, float param_lxd, float param_lyd, float param_Mt,
    float param_eta, float param_U0, float param_nts, float param_ictype,
    float* x, float* y, float* phi, float* psi,float* U);

void printCudaInfo();

// add function for easy retrieving params
template<class T>
T get(std::stringstream& ss) 
{
    T t; 
    ss<<t; 
    if (!ss) // if it failed to read
        throw std::runtime_error("could not parse parameter");
    return t;
}

void getParam(std::string lineText, std::string key, float& param){
    std::stringstream iss(lineText);
    std::string word;
    while (iss >> word){
        //std::cout << word << std::endl;
        std::string myKey=key;
        if(word!=myKey) continue;
        iss>>word;
        std::string equalSign="=";
        if(word!=equalSign) continue;
        iss>>word;
        param=std::stof(word);
        
    }
}

int main(int argc, char** argv)
{
    // All the variables should be float (except for nx, ny, Mt, nts, ictype, which should be integer)

    // step 1 (input): read or calculate  parameters from "input"
    // and print out information: lxd, nx, ny, Mt
    // Create a text string, which is used to output the text file
    char* fileName=argv[1];
    std::string lineText;

    std::ifstream parseFile(fileName);
    float param_G;
    
    float param_R; //= 50                          # pulling speed           um/s
    float param_delta; //= 0.02                    # strength of the surface tension anisotropy         
    float param_k;// = 0.14                        # interface solute partition coefficient
    float param_c_infm;// = 1.519                  # shift in melting temperature     K
    float param_Dl;// = 3000                       # liquid diffusion coefficient      um**2/s
    float param_d0;// = 0.02572                    # capillary length -- associated with GT coefficient   um
    float param_W0;// = 0.9375  
    float param_lT;// = c_infm*( 1.0/k-1 )/G       # thermal length           um
    float param_lamd;// = 5*np.sqrt(2)/8*W0/d0     # coupling constant
    float param_tau0;// = 0.6267*lamd*W0**2/Dl
    float param_c_infty;// = 2.45e-3
    float param_R_tilde;// = R*tau0/W0
    float param_Dl_tilde;// = Dl*tau0/W0**2
    float param_lT_tilde;// = lT/W0
    
    float param_eps;// = 1e-8                      	# divide-by-zero treatment
    float param_alpha0;// = 0                    	# misorientation angle in degree
    
    float param_dx;// = 1.2                            # mesh width
    float param_dt;// = 0.8*(dx)**2/(4*Dl_tilde)       # time step size for forward euler

    float param_asp_ratio;// = 4                  	# aspect ratio
    float param_nx;// = 128            		# number of cells in x, number of grids is nx+1
    float param_ny;// = asp_ratio*nx


    float param_lxd;// = dx*W0*nx                     # horizontal length in micron
    float param_lyd;// = asp_ratio*lxd

    float param_Mt;// = 10000                      	# total  number of time steps

    float param_eta;// = 0.0                		# magnitude of noise

    float param_U0;// = -0.3                		# initial value for U, -1 < U0 < 0
    float param_nts;// = 10				# number snapshots to save, Mt/nts must be int
    float param_ictype;// = 0                   	# initial condtion: 0 for semi-circular, 1 for planar interface, 2 for sum of sines

    while (parseFile.good()){
        std::getline (parseFile, lineText);
        // Output the text from the file
        getParam(lineText, "G", param_G);
        getParam(lineText, "R", param_R); 
        getParam(lineText, "delta", param_delta); 
        getParam(lineText, "k", param_k); 
        getParam(lineText, "c_infm", param_c_infm); 
        getParam(lineText, "Dl", param_Dl); 
        getParam(lineText, "d0", param_d0); 
        getParam(lineText, "W0", param_W0);  
        getParam(lineText, "c_infty", param_c_infty);
        getParam(lineText, "eps", param_eps);
        getParam(lineText, "alpha0", param_alpha0);
        getParam(lineText, "dx", param_dx);
        getParam(lineText, "asp_ratio", param_asp_ratio);
        getParam(lineText, "nx", param_nx);
        getParam(lineText, "Mt", param_Mt);
        getParam(lineText, "eta", param_eta);
        getParam(lineText, "U0", param_U0);
        getParam(lineText, "nts", param_nts);
        getParam(lineText, "ictype", param_ictype);
    }
    
    std::cout<<"G = "<<param_G<<std::endl;
    std::cout<<"R = "<<param_R<<std::endl;
    std::cout<<"delta = "<<param_delta<<std::endl;
    std::cout<<"k = "<<param_k<<std::endl;
    std::cout<<"c_infm = "<<param_c_infm<<std::endl;
    std::cout<<"Dl = "<<param_Dl<<std::endl;
    std::cout<<"d0 = "<<param_d0<<std::endl;
    std::cout<<"W0 = "<<param_W0<<std::endl;
    std::cout<<"c_infty = "<<param_c_infty<<std::endl;
    std::cout<<"eps = "<<param_eps<<std::endl;
    std::cout<<"alpha0 = "<<param_alpha0<<std::endl;
    std::cout<<"dx = "<<param_dx<<std::endl;
    std::cout<<"asp_ratio = "<<param_asp_ratio<<std::endl;
    std::cout<<"nx = "<<param_nx<<std::endl;
    std::cout<<"Mt = "<<param_Mt<<std::endl;
    std::cout<<"eta = "<<param_eta<<std::endl;
    std::cout<<"U0 = "<<param_U0<<std::endl;
    std::cout<<"nts = "<<param_nts<<std::endl;
    std::cout<<"ictype = "<<param_ictype<<std::endl;
    // Close the file
    parseFile.close();
    param_lT = param_c_infm*( 1.0/param_k-1 )/param_G;//       # thermal length           um
    param_lamd = 5*sqrt(2)/8*param_W0/param_d0;//     # coupling constant
    param_tau0 = pow(0.6267*param_lamd*param_W0,2)/param_Dl; //    # time scale  
    param_R_tilde = param_R*param_tau0/param_W0;
    param_Dl_tilde = pow(param_Dl*param_tau0/param_W0,2);
    param_lT_tilde = param_lT/param_W0;
    param_dt = pow(0.8*(param_dx),2)/(4*param_Dl_tilde);
    param_ny = param_asp_ratio*param_nx;
    param_lxd = param_dx*param_W0*param_nx; //                    # horizontal length in micron
    param_lyd = param_asp_ratio*param_lxd;
    std::cout<<"lT = "<<param_lT<<std::endl;
    std::cout<<"lamd = "<<param_lamd<<std::endl;
    std::cout<<"tau0 = "<<param_tau0<<std::endl;
    std::cout<<"R_tilde = "<<param_R_tilde<<std::endl;
    std::cout<<"Dl_tilde = "<<param_Dl_tilde<<std::endl;
    std::cout<<"lT_tilde = "<<param_lT_tilde<<std::endl;
    std::cout<<"dt = "<<param_dt<<std::endl;
    std::cout<<"ny = "<<param_ny<<std::endl;
    std::cout<<"lxd = "<<param_lxd<<std::endl;
    std::cout<<"lyd = "<<param_lyd<<std::endl;
    
    // step 2 (setup): pass the parameters to constant memory and 
    // allocate and initialize 1_D arrays on CPU/GPU  x: size nx+1, range [0,lxd], y: size ny+1, range [0, lyd]
    // you should get dx = dy = lxd/nx = lyd/ny

    // allocate 1_D arrays on CPU: psi, phi, U of size (nx+3)*(ny+3) -- these are for I/O
    // x and y would be allocate to the shared memory?
    
    int length_x=(int)param_nx+1;
    int length_y=(int)param_ny+1;
    float* x=(float*) malloc(length_x* sizeof(float));
    float* y=(float*) malloc(length_y* sizeof(float));

    // float* x = NULL;
    // float* y = NULL;
    // cudaMallocManaged((void**)&x, length_x* sizeof(float));
    // cudaMallocManaged((void**)&y, length_y* sizeof(float));

    // x
    for(int i=0; i<(int)param_nx+1; i++){
        x[i]=i*param_lxd/param_nx; 
    }

    std::cout<<"x= ";
    for(int i=0; i<(int)param_nx+1; i++){
        std::cout<<x[i]<<" ";
    }
    std::cout<<std::endl;

    // y
    for(int i=0; i<(int)param_ny+1; i++){
        y[i]=i*param_lyd/param_ny; 
    }

    std::cout<<"y= ";
    for(int i=0; i<(int)param_nx+1; i++){
        std::cout<<y[i]<<" ";
    }
    std::cout<<std::endl;

    int length=((int)param_nx+3) * ((int)param_ny+3);
    std::cout<<"length of psi, phi, U="<<length<<std::endl;
    float* psi=(float*) malloc(length* sizeof(float));
    float* phi=(float*) malloc(length* sizeof(float));
    float* U=(float*) malloc(length* sizeof(float));
    for(int i=0; i<length; i++){
        psi[i]=0.0;
        phi[i]=0.0;
        U[i]=0.0;
    }    

    setup(param_nx, param_ny, param_G, param_R,  param_delta,
         param_k,  param_c_infm,  param_Dl,  param_d0,  param_W0, 
         param_lT,  param_lamd,  param_tau0,  param_c_infty,  param_R_tilde,
         param_Dl_tilde,  param_lT_tilde,  param_eps,  param_alpha0,  param_dx, 
         param_dt,  param_asp_ratio,  param_lxd,  param_lyd,  param_Mt,
         param_eta,  param_U0,  param_nts,  param_ictype,
         x, y, phi, psi, U);
    // // allocate 1_D arrays on GPU: psi_old/psi_new, phi_old/phi_new, U_old/U_new, same size as before
    // float* psi_old = NULL;
    // float* psi_new = NULL;
    // float* U_old = NULL;
    // float* U_new = NULL;
    // float* phi_old = NULL;
    // float* phi_new = NULL;
    // cudaMalloc((void**)&psi_old, length* sizeof(float));
    // cudaMalloc((void**)&psi_new, length* sizeof(float));
    // cudaMalloc((void**)&U_old, length* sizeof(float));
    // cudaMalloc((void**)&U_new, length* sizeof(float));
    // cudaMalloc((void**)&phi_old, length* sizeof(float));
    // cudaMalloc((void**)&phi_new, length* sizeof(float));

    // // copy data from host to device to initialize old version
    // cudaMemcpy((void *)psi_old, (void *)psi, length* sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy((void *)U_old, (void *)U, length* sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy((void *)phi_old, (void *)phi, length* sizeof(float), cudaMemcpyHostToDevice);

    // step 3 (time marching): call the kernels Mt times
    
    // // step 4: save the psi, phi, U to a .mat file
    // MATFile *pmat = NULL;
    // mxArray *pMxArray = NULL;

    // const char *file = "output.mat";
    // printf("Creating file %s...\n\n", file);
    // pmat = matOpen(file, "w");

    // int M=(int)param_nx+3;
    // int N=(int)param_ny+3;
    // pMxArray = mxCreateDoubleMatrix(M, N, mxREAL);

    // mxSetData(pMxArray, psi_old);
    // matPutVariable(pmat, "psi", pMxArray);
    // mxSetData(pMxArray, phi_old);
    // matPutVariable(pmat, "phi", pMxArray);
    // mxSetData(pMxArray, U_old);
    // matPutVariable(pmat, "U", pMxArray);

    // // clean up before exit
    // mxDestroyArray(pMxArray);

    // if (matClose(pmat) != 0) {
    //     printf("Error closing file %s\n",file);
    //     return(EXIT_FAILURE);
    // }

    // printf("Done\n");

    return 0;
}
