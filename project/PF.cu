#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
// add thrust
#include <thrust/fill.h>
#include <thrust/device_vector.h>

using namespace std;
void printCudaInfo();
extern float toBW(int bytes, float sec);

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

};

__constant__ GlobalConstants cP;

// Device codes 

// boundary condition
// only use this function to access the boundary points, 
// other functions return at the boundary

__global__ void
set_BC(float* ps, float* ph, float* U, float* dpsi, int fnx, int fny){

  // find the location of boundary:
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // z=0, lx
  if (index<fnx) {
    int b_in = index+2*fnx;
    int t_out = index+(fny-1)*fnx;
    int t_in = index+(fny-3)*fnx;

    ps[index] = ps[b_in];
    ph[index] = ph[b_in];
    U[index] = U[b_in];
    dpsi[index] = dpsi[b_in];

    ps[t_out] = ps[t_in];
    ph[t_out] = ph[t_in];
    U[t_out] = U[t_in];
    dpsi[t_out] = dpsi[t_in];
  }
  if (index<fny){
    int l_out = index*fnx;
    int l_in = index*fnx + 2;
    int r_out = index*fnx + fnx -1;
    int r_in = index*fnx + fnx -3;
 
    ps[l_out] = ps[l_in];
    ph[l_out] = ph[l_in];
    U[l_out] = U[l_in];
    dpsi[l_out] = dpsi[l_in];
 
    ps[r_out] = ps[r_in];
    ph[r_out] = ph[r_in];
    U[r_out] = U[r_in];
    dpsi[r_out] = dpsi[r_in];
  }


}

// initialization
__global__ void
initialize(float* ps_old, float* ph_old, float* U_old, float* ps_new, float* ph_new, float* U_new
           , float* x, float* y, int fnx, int fny){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int j=C/fnx;
  int i=C-j*fnx;
  // when initialize, you need to consider C/F layout
  // if F layout, the 1D array has peroidicity of nx    
  // all the variables should be functions of x and y
  // size (nx+2)*(ny+2), x:nx, y:ny
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
    float xc = x[i];
    float yc = y[j];
    int cent = fnx/2;
    ps_old[C] = 5.625f - sqrtf( (xc-x[cent])*(xc-x[cent]) + yc*yc )/cP.W0 ;
    //if (C<1000){printf("ps %f\n",ps_old[C]);}
    ps_new[C] = ps_old[C];
    U_old[C] = cP.U0;
    U_new[C] = cP.U0;
    ph_old[C] = tanhf(ps_old[C]/cP.sqrt2);
    ph_new[C] = tanhf(ps_new[C]/cP.sqrt2);
  //  if (C<1000){printf("phi %f\n",ph_old[C]);} 
  }
}

// anisotropy functions
__device__ float
atheta(float ux, float uz){
  
   float ux2 = cP.cosa*ux + cP.sina*uz;
         ux2 = ux2*ux2;
   float uz2 = -cP.sina*ux + cP.cosa*uz;
         uz2 = uz2*uz2;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > cP.eps){
         return cP.a_s*( 1.0f + cP.epsilon*(ux2*ux2 + uz2*uz2) / MAG_sq2);}
   else {return 1.0f;}
}


__device__ float
aptheta(float ux, float uz){

   float uxr = cP.cosa*ux + cP.sina*uz;
   float ux2 = uxr*uxr;
   float uzr = -cP.sina*ux + cP.cosa*uz;
   float uz2 = uzr*uzr;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > cP.eps){
         return -cP.a_12*uxr*uzr*(ux2 - uz2) / MAG_sq2;}
   else {return 0.0f;}
}

// psi equation
__global__ void
rhs_psi(float* ps, float* ph, float* U, float* ps_new, float* ph_new, \
        float* y, float* dpsi, int fnx, int fny, int nt ){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int j=C/fnx; 
  int i=C-j*fnx;
  // if the points are at boundary, return
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
       // find the indices of the 8 neighbors for center
       //if (C==1000){printf("find");}
       int R=C+1;
       int L=C-1;
       int T=C+fnx;
       int B=C-fnx;
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ps's are defined on cell centers
        float psipjp=( ps[C] + ps[R] + ps[T] + ps[T+1] ) * 0.25f;
        float psipjm=( ps[C] + ps[R] + ps[B] + ps[B+1] ) * 0.25f;
        float psimjp=( ps[C] + ps[L] + ps[T-1] + ps[T] ) * 0.25f;
        float psimjm=( ps[C] + ps[L] + ps[B-1] + ps[B] ) * 0.25f;

        float phipjp=( ph[C] + ph[R] + ph[T] + ph[T+1] ) * 0.25f;
        float phipjm=( ph[C] + ph[R] + ph[B] + ph[B+1] ) * 0.25f;
        float phimjp=( ph[C] + ph[L] + ph[T-1] + ph[T] ) * 0.25f;
        float phimjm=( ph[C] + ph[L] + ph[B-1] + ph[B] ) * 0.25f;
        
        // ============================
        // right edge flux
        // ============================
        float psx = ps[R]-ps[C];
        float psz = psipjp - psipjm;
        float phx = ph[R]-ph[C];
        float phz = phipjp - phipjm;

        float A  = atheta( phx,phz);
        float Ap = aptheta(phx,phz);
        float JR = A * ( A*psx - Ap*psz );
        
        // ============================
        // left edge flux
        // ============================
        psx = ps[C]-ps[L];
        psz = psimjp - psimjm;
        phx = ph[C]-ph[L];
        phz = phimjp - phimjm; 

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JL = A * ( A*psx - Ap*psz );
        
        // ============================
        // top edge flux
        // ============================
        psx = psipjp - psimjp;
        psz = ps[T]-ps[C];
        phx = phipjp - phimjp;
        phz = ph[T]-ph[C];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JT = A * ( A*psz + Ap*psx );

        // ============================
        // bottom edge flux
        // ============================
        psx = psipjm - psimjm;
        psz = ps[C]-ps[B];
        phx = phipjm - phimjm;
        phz = ph[C]-ph[B];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JB = A * ( A*psz + Ap*psx );

         /*# =============================================================
        #
        # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
        #
        # =============================================================
        # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)*/
        float phxn = ( ph[R] - ph[L] ) * 0.5f;
        float phzn = ( ph[T] - ph[B] ) * 0.5f;
        float psxn = ( ps[R] - ps[L] ) * 0.5f;
        float pszn = ( ps[T] - ps[B] ) * 0.5f;

        float A2 = atheta(phxn,phzn);
        A2 = A2*A2;
        float gradps2 = (psxn)*(psxn) + (pszn)*(pszn);
        float extra =  -cP.sqrt2 * A2 * ph[C] * gradps2;

        /*# =============================================================
        #
        # 3. double well (transformed): sqrt2 * phi + nonlinear terms
        #
        # =============================================================*/

        float Up = (y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;

        float rhs_psi = ((JR-JL) + (JT-JB) + extra) * cP.hi*cP.hi + \
                   cP.sqrt2*ph[C] - cP.lamd*(1.0f-ph[C]*ph[C])*cP.sqrt2*(U[C] + Up);

        /*# =============================================================
        #
        # 4. dpsi/dt term
        #
        # =============================================================*/
        float tp = (1.0f-(1.0f-cP.k)*Up);
        float tau_psi;
        if (tp >= cP.k){tau_psi = tp*A2;}
               else {tau_psi = cP.k*A2;}
        
        dpsi[C] = rhs_psi / tau_psi; 
        
        ps_new[C] = ps[C] +  cP.dt * dpsi[C];
        ph_new[C] = tanhf(ps_new[C]/cP.sqrt2);
        //if (C==1000){printf("%f ",ph_new[C]);}
         }
} 

// U equation
__global__ void
rhs_U(float* U, float* U_new, float* ph, float* dpsi, int fnx, int fny ){

  int C = blockIdx.x * blockDim.x + threadIdx.x;
  int j=C/fnx;
  int i=C-j*fnx;
  // if the points are at boundary, return
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
        // find the indices of the 8 neighbors for center
        int R=C+1;
        int L=C-1;
        int T=C+fnx;
        int B=C-fnx;
        float hi = cP.hi;
        float Dl_tilde = cP.Dl_tilde;
        float k = cP.k;
        float nx,nz;
        float eps = cP.eps;
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ps's are defined on cell centers
        float phipjp=( ph[C] + ph[R] + ph[T] + ph[T+1] ) * 0.25f;
        float phipjm=( ph[C] + ph[R] + ph[B] + ph[B+1] ) * 0.25f;
        float phimjp=( ph[C] + ph[L] + ph[T-1] + ph[T] ) * 0.25f;
        float phimjm=( ph[C] + ph[L] + ph[B-1] + ph[B] ) * 0.25f;

        float jat    = 0.5f*(1.0f+(1.0f-k)*U[C])*(1.0f-ph[C]*ph[C])*dpsi[C];
        /*# ============================
        # right edge flux (i+1/2, j)
        # ============================*/
        float phx = ph[R]-ph[C];
        float phz = phipjp - phipjm;
        float phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else {nx = 0.0f;}
        
        float jat_ip = 0.5f*(1.0f+(1.0f-k)*U[R])*(1.0f-ph[R]*ph[R])*dpsi[R];	
        float UR = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[R])*(U[R]-U[C]) + 0.5f*(jat + jat_ip)*nx;
    	 
    	 
        /* ============================
        # left edge flux (i-1/2, j)
        # ============================*/
        phx = ph[C]-ph[L];
        phz = phimjp - phimjm;
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else {nx = 0.0f;}
        
        float jat_im = 0.5f*(1.0f+(1.0f-k)*U[L])*(1.0f-ph[L]*ph[L])*dpsi[L];
        float UL = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[L])*(U[C]-U[L]) + 0.5f*(jat + jat_im)*nx;
    	 
    	 
        /*# ============================
        # top edge flux (i, j+1/2)
        # ============================*/     
        phx = phipjp - phimjp;
        phz = ph[T]-ph[C];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else {nz = 0.0f;}    	
  
        float jat_jp = 0.5f*(1.0f+(1.0f-k)*U[T])*(1.0f-ph[T]*ph[T])*dpsi[T];      
        
        float UT = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[T])*(U[T]-U[C]) + 0.5f*(jat + jat_jp)*nz;
    	 
    	 
        /*# ============================
        # bottom edge flux (i, j-1/2)
        # ============================*/  
        phx = phipjm - phimjm;
        phz = ph[C]-ph[B];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else {nz = 0.0f;} 

        float jat_jm = 0.5f*(1.0f+(1.0f-k)*U[B])*(1.0f-ph[B]*ph[B])*dpsi[B];              
        float UB = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[B])*(U[C]-U[B]) + 0.5f*(jat + jat_jm)*nz;
        
        float rhs_U = ( (UR-UL) + (UT-UB) ) * hi + cP.sqrt2 * jat;
        float tau_U = (1.0f+cP.k) - (1.0f-cP.k)*ph[C];

        U_new[C] = U[C] + cP.dt * ( rhs_U / tau_U );

       }
}

__global__ void
getMeanY(float* phi, float* meanY, int fnx, int fny){
  
  int index=blockIdx.x * blockDim.x + threadIdx.x;
  if(index>=fnx){
    return;
  }
  float sum=0;
  for(int i=0; i<fny; i++){
    sum+=phi[fnx*i+index];
  }
  meanY[index]=sum/fny;
  //printf("%f ", meanY[index]);
}

__global__ void
getMeanX(float* phi, float* meanX, int fnx, int fny){
  
  int index=blockIdx.x * blockDim.x + threadIdx.x;
  if(index>=fny){
    return;
  }
  float sum=0;
  for(int i=0; i<fnx; i++){
    sum+=phi[fnx*index+i];
  }
  meanX[index]=sum/fnx;
  //printf("%f ", meanY[index]);
}

__global__ void
setZero(float* array, int length){
  
  int index=blockIdx.x * blockDim.x + threadIdx.x;
  if(index>=length){
    return;
  }
  array[index]=0;
  //printf("%f ", meanY[index]);
}

__global__ void
getSumX_parallel(float* phi, float* meanX, int fnx, int fny){
  
  int indexX=blockIdx.x * blockDim.x + threadIdx.x;
  int indexY=blockIdx.y * blockDim.y + threadIdx.y;
  if(indexX>=fnx || indexY>= fny){
    return;
  }
  atomicAdd(meanX+indexY, phi[indexY*fnx+indexX]);
  //printf("%f ", meanY[index]);
}

__global__ void
getSumY_parallel(float* phi, float* meanY, int fnx, int fny, int boxNum, int* boxLeftBound, int* boxRightBound, int* boxUpperBound, int* boxLowerBound, int* startIndex){
  
  int indexX=blockIdx.x * blockDim.x + threadIdx.x;
  int indexY=blockIdx.y * blockDim.y + threadIdx.y;
  if(indexX>=fnx || indexY>= fny){
    return;
  }
  for(int i=0; i<boxNum; i++){
    //in the i th box
    if(indexX>=boxLeftBound[i] && indexX<=boxRightBound[i] && indexY>=boxLowerBound[i] && indexY<=boxUpperBound[i]){
      atomicAdd(meanY+startIndex[i]+indexX-boxLeftBound[i], phi[indexY*fnx+indexX]);
    }
  
  //printf("%f ", meanY[index]);
  }
}

__global__ void
divide(float* array, int length, int* num, int* startIndex, int boxNum){
  
  int index=blockIdx.x * blockDim.x + threadIdx.x;
  if(index>=length){
    return;
  }
  for(int i=0; i<boxNum; i++){
    //in the ith box's partition
    if(index>=startIndex[i] && index< startIndex[i+1])
      array[index]=array[index]/num[i];
  }
  //printf("%f ", meanY[index]);
}

__global__ void
getTip(float* meanX, int* tipPos_device, int arrayIndex, int fny){
  int i=fny-1;
  //int epsilon=0.0000001f;
  while(i>=0 && !(meanX[i]>(-1.0f))){
    i--;
  }
  tipPos_device[arrayIndex]=i;
  //printf("tipPos=%d\n",tipPos_device[arrayIndex]);
}

__global__ void
getCell(float* meanY, float* cellNum, int arrayIndex, int fnx){
  bool positive=meanY[0]>0;
  int crossNum=0;
  for(int i=0; i<fnx; i++){
    if((positive && meanY[i]<0) || (!positive && meanY[i]>0)){
      positive=!positive;
      crossNum++;
    }
  }
  cellNum[arrayIndex]=crossNum/2;
}

__global__ void reduce0(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];
  // perform first level of reduction,
  // reading from global memory, writing to shared memory

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
  __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void
getC(float* C, float* U, float* phi, int fnx, int fny){
  int index=blockIdx.x * blockDim.x + threadIdx.x;
  if(index>=fny*fny){
    return;
  }
  C[index]=cP.c_infm*(U[index]*(1-cP.k)+cP.k)*(1-phi[index]+cP.k*(1+phi[index]))/(2*cP.k);
}
__global__ void
simpleSum(float* array, float* out_array, int out_index, int length){
  int index=blockIdx.x * blockDim.x + threadIdx.x;
  if(index>=length){
    return;
  }
  atomicAdd(out_array+out_index, array[index]);
}


void setup(GlobalConstants params, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U){
  // we should have already pass all the data structure in by this time
  // move those data onto device
  printCudaInfo();
  float* x_device;// = NULL;
  float* y_device;// = NULL;

  float* psi_old;// = NULL;
  float* psi_new;// = NULL;
  float* U_old;// = NULL;
  float* U_new;// = NULL;
  float* phi_old;// = NULL;
  float* phi_new;// = NULL;
  float* dpsi;// = NULL;
  // allocate x, y, phi, psi, U related params
  int length = fnx*fny;

  cudaMalloc((void **)&x_device, sizeof(float) * fnx);
  cudaMalloc((void **)&y_device, sizeof(float) * fny);

  cudaMalloc((void **)&phi_old,  sizeof(float) * length);
  cudaMalloc((void **)&psi_old,  sizeof(float) * length);
  cudaMalloc((void **)&U_old,    sizeof(float) * length);
  cudaMalloc((void **)&phi_new,  sizeof(float) * length);
  cudaMalloc((void **)&psi_new,  sizeof(float) * length);
  cudaMalloc((void **)&U_new,    sizeof(float) * length);
  cudaMalloc((void **)&dpsi,    sizeof(float) * length);

  cudaMemcpy(x_device, x, sizeof(float) * fnx, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y, sizeof(float) * fny, cudaMemcpyHostToDevice);
  cudaMemcpy(psi_old, psi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(phi_old, phi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(U_old, U, sizeof(float) * length, cudaMemcpyHostToDevice);

  // pass all the read-only params into global constant
  cudaMemcpyToSymbol(cP, &params, sizeof(GlobalConstants));

   int blocksize_1d = 128;
   int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
   int num_block_2d = (fnx*fny+blocksize_2d-1)/blocksize_2d;
   int num_block_1d = (fnx+fny+blocksize_1d-1)/blocksize_1d;
   printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 
   initialize<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, U_new, x_device, y_device, fnx, fny);
   set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_new, dpsi, fnx, fny);
   set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_old, dpsi, fnx, fny);
   cudaDeviceSynchronize();
   double startTime = CycleTimer::currentSeconds();
   for (int kt=0; kt<params.Mt/2; kt++){
   //  printf("time step %d\n",kt);
     rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, y_device, dpsi, fnx, fny, 2*kt );
     //cudaDeviceSynchronize();
     set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, fnx, fny);
     //cudaDeviceSynchronize();
     rhs_U<<< num_block_2d, blocksize_2d >>>(U_old, U_new, phi_new, dpsi, fnx, fny);

     //cudaDeviceSynchronize();
     rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_new, phi_new, U_new, psi_old, phi_old, y_device, dpsi, fnx, fny, 2*kt+1 );
     //cudaDeviceSynchronize();
     set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, fnx, fny);
     //cudaDeviceSynchronize();
     rhs_U<<< num_block_2d, blocksize_2d >>>(U_new, U_old, phi_old, dpsi, fnx, fny);
     //cudaDeviceSynchronize();
   }
   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", params.Mt, endTime-startTime);
   cudaMemcpy(psi, psi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(phi, phi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(U, U_old, length * sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(x_device); cudaFree(y_device);
  cudaFree(psi_old); cudaFree(psi_new);
  cudaFree(phi_old); cudaFree(phi_new);
  cudaFree(U_old); cudaFree(U_new);
  cudaFree(dpsi);  


}

void my_setup(GlobalConstants params, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U, int* tipPos, float* cellNum, float* asc, int boxNum, int* boxSizeX, int*boxSizeY, int* boxPosX, int* boxPosY){
  // we should have already pass all the data structure in by this time
  // move those data onto device
  printCudaInfo();
  float* x_device;// = NULL;
  float* y_device;// = NULL;

  float* psi_old;// = NULL;
  float* psi_new;// = NULL;
  float* U_old;// = NULL;
  float* U_new;// = NULL;
  float* phi_old;// = NULL;
  float* phi_new;// = NULL;
  float* dpsi;// = NULL;
  // allocate x, y, phi, psi, U related params
  int length = fnx*fny;

  cudaMalloc((void **)&x_device, sizeof(float) * fnx);
  cudaMalloc((void **)&y_device, sizeof(float) * fny);

  cudaMalloc((void **)&phi_old,  sizeof(float) * length);
  cudaMalloc((void **)&psi_old,  sizeof(float) * length);
  cudaMalloc((void **)&U_old,    sizeof(float) * length);
  cudaMalloc((void **)&phi_new,  sizeof(float) * length);
  cudaMalloc((void **)&psi_new,  sizeof(float) * length);
  cudaMalloc((void **)&U_new,    sizeof(float) * length);
  cudaMalloc((void **)&dpsi,    sizeof(float) * length);

  cudaMemcpy(x_device, x, sizeof(float) * fnx, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y, sizeof(float) * fny, cudaMemcpyHostToDevice);
  cudaMemcpy(psi_old, psi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(phi_old, phi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(U_old, U, sizeof(float) * length, cudaMemcpyHostToDevice);

  // pass all the read-only params into global constant
  cudaMemcpyToSymbol(cP, &params, sizeof(GlobalConstants));

  printf("fnx=%d, fny=%d\n", fnx, fny);
  int* tipPos_device;
  cudaMalloc((void**)&tipPos_device, sizeof(int)*params.Mt*boxNum);
  float* cellNum_device;
  cudaMalloc((void**)&cellNum_device, sizeof(float)*params.Mt*boxNum);
  float* asc_device;
  cudaMalloc((void**)&asc_device, sizeof(float)*params.Mt*boxNum);


  float* meanY;
  int meanYLength=0;
  for(int i=0; i<boxNum; i++){
    meanYLength+=boxSizeX[i];
  }
  cudaMalloc((void**)&meanY, sizeof(float)*meanYLength);
  float meanY_host[meanYLength];

  float* meanX;
  int meanXLength=0;
  for(int i=0; i<boxNum; i++){
    meanXLength+=boxSizeY[i];
  }
  cudaMalloc((void**)&meanX, sizeof(float)*meanXLength);
  float meanX_host[meanXLength];

  // float* meanX_1;
  // cudaMalloc((void**)&meanX_1, sizeof(float)*fny);
  // float* meanY_1;
  // cudaMalloc((void**)&meanY_1, sizeof(float)*fnx);

  

  float* C;
  cudaMalloc((void**)&C, sizeof(float)*fnx*fny);
  float C_host[fnx*fny];



   int blocksize_1d = 128;
   int blocksize_2d = 128;  // seems reduce the block size makes it a little faster, but around 128 is okay.
   int num_block_2d = (fnx*fny+blocksize_2d-1)/blocksize_2d;
   int num_block_1d = (fnx+fny+blocksize_1d-1)/blocksize_1d;
   printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 
   initialize<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, U_new, x_device, y_device, fnx, fny);
   set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_new, dpsi, fnx, fny);
   set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_old, dpsi, fnx, fny);
   cudaDeviceSynchronize();
   double startTime = CycleTimer::currentSeconds();
   int my_blockSize=128;
   int my_numBlockX=(fnx+my_blockSize-1)/my_blockSize;
   int my_numBlockY=(fny+my_blockSize-1)/my_blockSize;

   dim3 my_blockSize2d_y(1,128);
   dim3 my_numBlock2d_y((fnx+my_blockSize2d_y.x-1)/my_blockSize2d_y.x, (fny+my_blockSize2d_y.y-1)/my_blockSize2d_y.y);
   dim3 my_blockSize2d_x(128,1);
   dim3 my_numBlock2d_x((fnx+my_blockSize2d_x.x-1)/my_blockSize2d_x.x, (fny+my_blockSize2d_x.y-1)/my_blockSize2d_x.y);

   int boxLeftBound[boxNum];
   int boxRightBound[boxNum];
   int boxUpperBound[boxNum];
   int boxLowerBound[boxNum];
   for(int i=0; i<boxNum; i++){
     boxLeftBound[i]=boxPosX[i];
     boxRightBound[i]=boxPosX[i]+boxSizeX[i]-1;
     boxLowerBound[i]=boxPosY[i];
     boxUpperBound[i]=boxPosY[i]+boxSizeY[i]-1;
   }
   int* boxLeftBound_device;
   int* boxRightBound_device;
   int* boxUpperBound_device;
   int* boxLowerBound_device;
   cudaMalloc((void**)&boxLeftBound_device, sizeof(int)*boxNum);
   cudaMalloc((void**)&boxRightBound_device, sizeof(int)*boxNum);
   cudaMalloc((void**)&boxUpperBound_device, sizeof(int)*boxNum);
   cudaMalloc((void**)&boxLowerBound_device, sizeof(int)*boxNum);
   cudaMemcpy(boxLeftBound_device, boxLeftBound, sizeof(int)*boxNum, cudaMemcpyHostToDevice);
   cudaMemcpy(boxRightBound_device, boxRightBound, sizeof(int)*boxNum, cudaMemcpyHostToDevice);
   cudaMemcpy(boxUpperBound_device, boxUpperBound, sizeof(int)*boxNum, cudaMemcpyHostToDevice);
   cudaMemcpy(boxLowerBound_device, boxLowerBound, sizeof(int)*boxNum, cudaMemcpyHostToDevice);
   int meanXStartIndex[boxNum+1];
   int meanYStartIndex[boxNum+1];
   for(int i=0; i<boxNum+1; i++){
     if(i==0){
       meanXStartIndex[i]=0;
       meanYStartIndex[i]=0;
     }
     else {
       meanXStartIndex[i]=meanXStartIndex[i-1]+boxSizeY[i-1];
       meanYStartIndex[i]=meanYStartIndex[i-1]+boxSizeX[i-1];
     }
   }
   int* meanXStartIndex_device;
   int* meanYStartIndex_device;
   cudaMalloc((void**)&meanXStartIndex_device, sizeof(int)*(boxNum+1));
   cudaMalloc((void**)&meanYStartIndex_device, sizeof(int)*(boxNum+1));
   cudaMemcpy(meanXStartIndex_device, meanXStartIndex, sizeof(int)*(boxNum+1), cudaMemcpyHostToDevice);
   cudaMemcpy(meanYStartIndex_device, meanYStartIndex, sizeof(int)*(boxNum+1), cudaMemcpyHostToDevice);
   
    double totalTime1=0;
    double totalTime2=0;
    double totalTime3=0;
    double totalTime4=0;
    double totalTime5=0;

   for (int kt=0; kt<params.Mt/2; kt++){
      double time1 = CycleTimer::currentSeconds();
   //  printf("time step %d\n",kt);
      rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, y_device, dpsi, fnx, fny, 2*kt );
      //cudaDeviceSynchronize();
      set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, fnx, fny);
      //cudaDeviceSmeanYynchronize();
      rhs_U<<< num_block_2d, blocksize_2d >>>(U_old, U_new, phi_new, dpsi, fnx, fny);
      
      cudaDeviceSynchronize();
      double time2 = CycleTimer::currentSeconds();
      totalTime1+=(time2-time1);

      //getMeanY<<<my_numBlockX,my_blockSize>>>(phi_new, meanY, fnx, fny);
      setZero<<<(meanYLength+my_blockSize-1)/my_blockSize, my_blockSize>>>(meanY, meanYLength);
      getSumY_parallel<<<my_numBlock2d_x,my_blockSize2d_x>>>(phi_new, meanY, fnx, fny, boxNum, boxLeftBound_device, boxRightBound_device, boxUpperBound_device, boxLowerBound_device, meanYStartIndex_device);
      divide<<<(meanYLength+my_blockSize-1)/my_blockSize, my_blockSize>>>(meanY, meanYLength, boxSizeY, meanYStartIndex_device, boxNum);

      cudaDeviceSynchronize();
      double time3 = CycleTimer::currentSeconds();
      totalTime2+=(time3-time2);

      //1d parallel
      //getMeanX<<<my_numBlockY,my_blockSize>>>(phi_new, meanX, fnx, fny);
      //2d parallel with atomicAdd
      setZero<<<num_block_2d, blocksize_2d>>>(meanX, fny);
      getSumX_parallel<<<my_numBlock2d_y,my_blockSize2d_y>>>(phi_new, meanX, fnx, fny);
      divide<<<num_block_2d, blocksize_2d>>>(meanX, fny, fnx);

      cudaDeviceSynchronize();
      double time4 = CycleTimer::currentSeconds();
      totalTime3+=(time4-time3);

      // cudaMemcpy(meanY_host, meanY, sizeof(float)*fnx, cudaMemcpyDeviceToHost);
      // for(int i=0; i<fnx; i++){
      //   printf("%f ", meanY_host[i]);
      // }
      // printf("\n");
      // cudaMemcpy(meanY_host, meanY_1, sizeof(float)*fnx, cudaMemcpyDeviceToHost);
      // for(int i=0; i<fnx; i++){
      //   printf("%f ", meanY_host[i]);
      // }
      // printf("\n");

      // cudaMemcpy(meanX_host, meanX, sizeof(float)*fny, cudaMemcpyDeviceToHost);
      // for(int i=0; i<fny; i++){
      //   printf("%f ", meanX_host[i]);
      // }
      // printf("\n");
      // cudaMemcpy(meanX_host, meanX_1, sizeof(float)*fny, cudaMemcpyDeviceToHost);
      // for(int i=0; i<fny; i++){
      //   printf("%f ", meanX_host[i]);
      // }
      // printf("\n");

      getTip<<<1,1>>>(meanX, tipPos_device, 2*kt, fny);
      getCell<<<1,1>>>(meanY, cellNum_device, 2*kt, fnx);

      cudaDeviceSynchronize();
      double time5 = CycleTimer::currentSeconds();
      totalTime4+=(time5-time4);

      getC<<<num_block_2d, blocksize_2d>>>(C, U_new, phi_new, fnx, fny);

      // cudaMemcpy(C_host, C, sizeof(float)* fnx* fny, cudaMemcpyDeviceToHost);
      // if(kt==0){
      //   for(int i=0; i<fny; i++){
      //     for(int j=0; j<fnx; j++){
      //       printf("%f ", C_host[i*fnx+j]);
      //     }
      //     printf("\n");
      //   }
      // }

      simpleSum<<<num_block_2d, blocksize_2d>>>(C, asc_device, 2*kt, length);

      cudaDeviceSynchronize();
      double time6 = CycleTimer::currentSeconds();
      totalTime5+=(time6-time5);
      

      //cudaDeviceSynchronize();
      rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_new, phi_new, U_new, psi_old, phi_old, y_device, dpsi, fnx, fny, 2*kt+1 );
      //cudaDeviceSynchronize();
      set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, fnx, fny);
      //cudaDeviceSynchronize();
      rhs_U<<< num_block_2d, blocksize_2d >>>(U_new, U_old, phi_old, dpsi, fnx, fny);
      //cudaDeviceSynchronize();
      //getMeanY<<<my_numBlockX,my_blockSize>>>(phi_old, meanY, fnx, fny);
      // setZero<<<num_block_2d, blocksize_2d>>>(meanY, fnx);
      // getSumY_parallel<<<my_numBlock2d_x,my_blockSize2d_x>>>(phi_old, meanY, fnx, fny);
      // divide<<<num_block_2d, blocksize_2d>>>(meanY, fnx, fny);
      // //getMeanX<<<my_numBlockY,my_blockSize>>>(phi_old, meanX, fnx, fny);
      // //2d parallel with atomicAdd
      // setZero<<<num_block_2d, blocksize_2d>>>(meanX, fny);
      // getSumX_parallel<<<my_numBlock2d_y,my_blockSize2d_y>>>(phi_old, meanX, fnx, fny);
      // divide<<<num_block_2d, blocksize_2d>>>(meanX, fny, fnx);

      // // cudaMemcpy(meanY_host, meanY, sizeof(float)*fnx, cudaMemcpyDeviceToHost);
      // // for(int i=0; i<fnx; i++){
      // //   printf("%f ", meanY_host[i]);
      // // }
      // // printf("\n");
      // // cudaMemcpy(meanX_host, meanX, sizeof(float)*fny, cudaMemcpyDeviceToHost);
      // // for(int i=0; i<fny; i++){
      // //   printf("%f ", meanX_host[i]);
      // // }
      // // printf("\n");

      // getTip<<<1,1>>>(meanX, tipPos_device, 2*kt+1, fny);
      // getCell<<<1,1>>>(meanY, cellNum_device, 2*kt+1, fnx);
      // getC<<<num_block_2d, blocksize_2d>>>(C, U_old, phi_old, fnx, fny);
      // simpleSum<<<num_block_2d, blocksize_2d>>>(C, asc_device, 2*kt+1, length);
    }
    divide<<<(params.Mt+127)/128, 128>>>(asc_device, params.Mt, fnx*fny);

      cudaMemcpy(tipPos, tipPos_device, sizeof(int)*params.Mt, cudaMemcpyDeviceToHost);
      printf("tipPos:");
      for(int i=0; i<params.Mt; i++){
        printf("%d ", tipPos[i]);
      }
      printf("\n");

      cudaMemcpy(cellNum, cellNum_device, sizeof(float)*params.Mt, cudaMemcpyDeviceToHost);
      printf("cellNum:");
      for(int i=0; i<params.Mt; i++){
        printf("%f ", cellNum[i]);
      }
      printf("\n");

      cudaMemcpy(asc, asc_device, sizeof(float)*params.Mt, cudaMemcpyDeviceToHost);
      printf("asc:");
      for(int i=0; i<params.Mt; i++){
        printf("%f ", asc[i]);
      }
      printf("\n");


   cudaDeviceSynchronize();
   double endTime = CycleTimer::currentSeconds();
   printf("time for %d iterations: %f s\n", params.Mt, endTime-startTime);

   printf("times in each iteration: %f, %f, %f, %f, %f\n", 
   totalTime1*2/params.Mt, totalTime2*2/params.Mt, totalTime3*2/params.Mt, totalTime4*2/params.Mt, totalTime5*2/params.Mt);

   cudaMemcpy(psi, psi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(phi, phi_old, length * sizeof(float),cudaMemcpyDeviceToHost);
   cudaMemcpy(U, U_old, length * sizeof(float),cudaMemcpyDeviceToHost);
  //  for(int i=0; i<fny; i++){
  //    for(int j=0; j<fnx; j++){
  //      printf("%f ", phi[i*fnx+j]);
  //    }
  //    printf("\n");
  //  }

  cudaFree(x_device); cudaFree(y_device);
  cudaFree(psi_old); cudaFree(psi_new);
  cudaFree(phi_old); cudaFree(phi_new);
  cudaFree(U_old); cudaFree(U_new);
  cudaFree(dpsi);  


}

/*
void time_marching(GlobalConstants params, int fnx, int fny){

   // initialize or load

   int blocksize_1d = 256;
   int blocksize_2d = 512;
   int num_block_2d = (fnx*fny+blocksize_2d-1)/blocksize_2d;
   int num_block_1d = (fnx+fny+blocksize_1d-1)/blocksize_1d;

   initialize<<< num_block_2d, blocksize_2d >>>(ps_old, ph_old, U_old, ps_new, ph_new, U_new, x_device, y_device, fnx, fny);
   

   for (int kt=0; kt<params.Mt/2; kt++){

     rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, y_device, dpsi, fnx, fny, 2*kt ); 
     set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, fnx, fny);
     rhs_U<<< num_block_2d, blocksize_2d >>>(U_old, U_new, phi_new, dpsi);


     rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_new, phi_new, U_new, psi_old, phi_old, y_device, dpsi, fnx, fny, 2*kt+1 ); 
     set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, fnx, fny);
     rhs_U<<< num_block_2d, blocksize_2d >>>(U_new, U_old, phi_old, dpsi);


   }

   
 
}*/




void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
