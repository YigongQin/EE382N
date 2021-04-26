#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <map>
#include<functional>

#include "CycleTimer.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "/home1/apps/matlab/2020b/extern/include/mat.h"
// #include <mat.h> 
using namespace std;



struct GlobalConstants {
  int nx;
  int ny;
  int Mt;
  int nts; 
  int ictype;
  float G;
  float R;
  float delta;
  float k;
  float c_infm;
  float Dl;
  float d0;
  float W0;
  float lT;
  float lamd; 
  float tau0;
  float c_infty; 
  float R_tilde;
  float Dl_tilde; 
  float lT_tilde; 
  float eps; 
  float alpha0; 
  float dx; 
  float dt; 
  float asp_ratio; 
  float lxd;
  float lx; 
  float lyd; 
  float eta; 
  float U0; 
  // parameters that are not in the input file
  float hi;
  float cosa;
  float sina;
  float sqrt2;
  float a_s;
  float epsilon;
  float a_12;

  float Ti;
  int ha_wd;
  int Mnx;
  int Mny;
  int Mnt;
  float xmin;
  float ymin;
  
};

struct params_MPI{

    int rank;
    int px;
    int py;
    int nproc;
    int nprocx;
    int nprocy;
    int nx_loc;
    int ny_loc;
};


void setup(MPI_Comm comm, params_MPI pM, GlobalConstants params, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U);


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

void matread(std::string matfile, std::string key, std::vector<double>& v)
{
    // open MAT-file
    MATFile *pmat = matOpen(matfile, "r");
    if (pmat == NULL) return;

    // extract the specified variable
    mxArray *arr = matGetVariable(pmat, key);
    if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {
        // copy data
        mwSize num = mxGetNumberOfElements(arr);
        double *pr = mxGetPr(arr);
        if (pr != NULL) {
            v.reserve(num); //is faster than resize :-)
            v.assign(pr, pr+num);
        }
    }

    // cleanup
    mxDestroyArray(arr);
    matClose(pmat);
}



int main(int argc, char** argv)
{
    // All the variables should be float (except for nx, ny, Mt, nts, ictype, which should be integer)

    // step 1 (input): read or calculate  parameters from "input"
    // and print out information: lxd, nx, ny, Mt
    // Create a text string, which is used to output the text file
    params_MPI pM;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &pM.rank);
    MPI_Comm_size(comm, &pM.nproc);


     
    //if rank == 0: print('GPUs on this node', cuda.gpus)
    //printf('device id',gpu_name,'host id',rank );

    pM.nprocx = (int) ceil(sqrt(pM.nproc));
    pM.nprocy = (int) ceil(pM.nproc/pM.nprocx);


    pM.px = pM.rank%pM.nprocx;           //# x id of current processor   [0:nprocx]
    pM.py = pM.rank/pM.nprocx; //# y id of current processor  [0:nprocy]
    printf("px, %d, py, %d, for rank %d \n",pM.px, pM.py, pM.rank);

    if (pM.rank ==0){ printf("total/x/y processors %d, %d, %d\n", pM.nproc, pM.nprocx, pM.nprocy);}



    char* fileName=argv[1];
    std::string lineText;
    std::string matfile=argv[2];

    std::ifstream parseFile(fileName);
    float nx;
    float Mt;
    float nts;
    float ictype;
    float ha_wd;
    GlobalConstants params;
    while (parseFile.good()){
        std::getline (parseFile, lineText);
        // Output the text from the file
        getParam(lineText, "G", params.G);
        getParam(lineText, "R", params.R); 
        getParam(lineText, "delta", params.delta); 
        getParam(lineText, "k", params.k); 
        getParam(lineText, "c_infm", params.c_infm); 
        getParam(lineText, "Dl", params.Dl); 
        getParam(lineText, "d0", params.d0); 
        getParam(lineText, "W0", params.W0);  
        getParam(lineText, "c_infty", params.c_infty);
        getParam(lineText, "eps", params.eps);
        getParam(lineText, "alpha0", params.alpha0);
        getParam(lineText, "dx", params.dx);
        getParam(lineText, "asp_ratio", params.asp_ratio);
        getParam(lineText, "nx", nx);
        params.nx = (int)nx;
        getParam(lineText, "Mt", Mt);
        params.Mt = (int)Mt;
        getParam(lineText, "eta", params.eta);
        getParam(lineText, "U0", params.U0);
        getParam(lineText, "nts", nts);
        params.nts = (int)nts;
        getParam(lineText, "ictype", ictype);
        params.ictype = (int)ictype;

        // new multiple
        getParam(lineText, "ha_wd", ha_wd);
        params.ha_wd = (int)ha_wd;
        getParam(lineText, "xmin", params.xmin);
        getParam(lineText, "ymin", params.ymin);
        
    }
    

    // Close the file
    parseFile.close();
    
    // calculate the parameters
    params.lT = params.c_infm*( 1.0/params.k-1 )/params.G;//       # thermal length           um
    params.lamd = 5*sqrt(2.0)/8*params.W0/params.d0;//     # coupling constant
    params.tau0 = 0.6267*params.lamd*params.W0*params.W0/params.Dl; //    # time scale  
    params.R_tilde = params.R*params.tau0/params.W0;
    params.Dl_tilde = params.Dl*params.tau0/pow(params.W0,2);
    params.lT_tilde = params.lT/params.W0;
    params.dt = 0.8*pow(params.dx,2)/(4*params.Dl_tilde);
    params.ny = (int)params.asp_ratio*params.nx;
    params.lxd = params.dx*params.W0*params.nx; //                    # horizontal length in micron
    params.lyd = params.asp_ratio*params.lxd;
    params.hi = 1.0/params.dx;
    params.cosa = cos(params.alpha0/180*M_PI);
    params.sina = sin(params.alpha0/180*M_PI);
    params.sqrt2 = sqrt(2.0);
    params.a_s = 1 - 3.0*params.delta;
    params.epsilon = 4.0*params.delta/params.a_s;
    params.a_12 = 4.0*params.a_s*params.epsilon;
   
    if (pM.rank==0){ 
    std::cout<<"G = "<<params.G<<std::endl;
    std::cout<<"R = "<<params.R<<std::endl;
    std::cout<<"delta = "<<params.delta<<std::endl;
    std::cout<<"k = "<<params.k<<std::endl;
    std::cout<<"c_infm = "<<params.c_infm<<std::endl;
    std::cout<<"Dl = "<<params.Dl<<std::endl;
    std::cout<<"d0 = "<<params.d0<<std::endl;
    std::cout<<"W0 = "<<params.W0<<std::endl;
    std::cout<<"c_infty = "<<params.c_infty<<std::endl;
    std::cout<<"eps = "<<params.eps<<std::endl;
    std::cout<<"alpha0 = "<<params.alpha0<<std::endl;
    std::cout<<"dx = "<<params.dx<<std::endl;
    std::cout<<"asp_ratio = "<<params.asp_ratio<<std::endl;
    std::cout<<"nx = "<<params.nx<<std::endl;
    std::cout<<"Mt = "<<params.Mt<<std::endl;
    std::cout<<"eta = "<<params.eta<<std::endl;
    std::cout<<"U0 = "<<params.U0<<std::endl;
    std::cout<<"nts = "<<params.nts<<std::endl;
    std::cout<<"ictype = "<<params.ictype<<std::endl;
    std::cout<<"lT = "<<params.lT<<std::endl;
    std::cout<<"lamd = "<<params.lamd<<std::endl;
    std::cout<<"tau0 = "<<params.tau0<<std::endl;
    std::cout<<"R_tilde = "<<params.R_tilde<<std::endl;
    std::cout<<"Dl_tilde = "<<params.Dl_tilde<<std::endl;
    std::cout<<"lT_tilde = "<<params.lT_tilde<<std::endl;
    std::cout<<"dt = "<<params.dt<<std::endl;
    std::cout<<"ny = "<<params.ny<<std::endl;
    std::cout<<"lxd = "<<params.lxd<<std::endl;
    std::cout<<"lyd = "<<params.lyd<<std::endl;

    }
    // step1 plus: read mat file from macrodata
    //
    //
 
    // step 2 (setup): pass the parameters to constant memory and 
    // allocate and initialize 1_D arrays on CPU/GPU  x: size nx+1, range [0,lxd], y: size ny+1, range [0, lyd]
    // you should get dx = dy = lxd/nx = lyd/ny

    // allocate 1_D arrays on CPU: psi, phi, U of size (nx+3)*(ny+3) -- these are for I/O
    // x and y would be allocate to the shared memory?

    
    
    //==============================
    // parameters depend on MPI
    pM.nx_loc = params.nx;
    pM.ny_loc = params.ny;
    
    float len_blockx = params.lxd/pM.nprocx;
    float len_blocky = params.lyd/pM.nprocy;
    
    float xmin_loc = params.xmin+ pM.px*len_blockx; 
    float ymin_loc = params.ymin+ pM.py*len_blocky; 

    if (pM.px==0){pM.nx_loc+=1;}
    else{xmin_loc+=params.lxd/params.nx;}
    
    if (pM.py==0){pM.ny_loc+=1;}
    else{ymin_loc+=params.lyd/params.ny;}

    int length_x = pM.nx_loc+2*params.ha_wd;
    int length_y = pM.ny_loc+2*params.ha_wd;
    float* x=(float*) malloc(length_x* sizeof(float));
    float* y=(float*) malloc(length_y* sizeof(float));


    for(int i=0; i<length_x; i++){
        x[i]=(i-params.ha_wd)*params.lxd/params.nx; 
    }

    for(int i=0; i<length_y; i++){
        y[i]=(i-params.ha_wd)*params.lyd/params.ny;
    }



//===================================

    std::cout<<"x= ";
    for(int i=0; i<length_x; i++){
        std::cout<<x[i]<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"y= ";
    for(int i=0; i<length_y; i++){
        std::cout<<y[i]<<" ";
    }
    std::cout<<std::endl;

    int length=length_x*length_y;
    std::cout<<"x length of psi, phi, U="<<length_x<<std::endl;
    std::cout<<"y length of psi, phi, U="<<length_y<<std::endl;
    std::cout<<"length of psi, phi, U="<<length<<std::endl;
    float* psi=(float*) malloc(length* sizeof(float));
    float* phi=(float*) malloc(length* sizeof(float));
    float* Uc=(float*) malloc(length* sizeof(float));
    for(int i=0; i<length; i++){
        psi[i]=0.0;
        phi[i]=0.0;
        Uc[i]=0.0;
    }
    
    //std::cout<<"y= ";
    //for(int i=0+length_y; i<2*length_y; i++){
    //    std::cout<<phi[i]<<" ";
    //}
    //std::cout<<std::endl;



    // step1 plus: read mat file from macrodata

    std::vector<double> v;
    matread(matfile, "x",v);
    for (std::size_t i=0; i<v.size(); ++i)
        {std::cout << v[i] << std::endl;}






    //setup(comm, pM, params, length_x, length_y, x, y, phi, psi, Uc);

    //std::cout<<"y= ";
    //for(int i=0+length_y; i<2*length_y; i++){
    //    std::cout<<Uc[i]<<" ";
    //}
    //std::cout<<std::endl;
    // step 3 (time marching): call the kernels Mt times
    double* phi_arr= new double[length];
    for(int i=0; i<length; i++){
         phi_arr[i] = (double) phi[i];
    } 
    // // step 4: save the psi, phi, U to a .mat file
    MATFile *pmatFile = NULL;
    mxArray *pMxArray = NULL;
    pmatFile = matOpen("MPI_data.mat","w");
    pMxArray = mxCreateDoubleMatrix(length_x, length_y, mxREAL);
    mxSetData(pMxArray, phi_arr);
    matPutVariable(pmatFile, "phi", pMxArray);
    //mxSetData(pMxArray, phi);
    //matPutVariable(pmatFile, "phi", pMxArray);
    //mxSetData(pMxArray, U);
    //matPutVariable(pmatFile, "U", pMxArray);
    matClose(pmatFile);
    delete[] phi;
    delete[] Uc;
    delete[] psi;
    return 0;
}
