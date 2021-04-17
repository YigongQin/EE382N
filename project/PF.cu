#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

using namespace std;

extern float toBW(int bytes, float sec);

struct GlobalConstants {
  float param_nx;
  float param_ny;
  float param_G;
  float param_R;
  float param_delta;
  float param_k;
  float param_c_infm;
  float param_Dl;
  float param_d0;
  float param_W0;
  float param_lT;
  float param_lamd; 
  float param_tau0;
  float param_c_infty; 
  float param_R_tilde;
  float param_Dl_tilde; 
  float param_lT_tilde; 
  float param_eps; 
  float param_alpha0; 
  float param_dx; 
  float param_dt; 
  float param_asp_ratio; 
  float param_lxd; 
  float param_lyd; 
  float param_Mt;
  float param_eta; 
  float param_U0; 
  float param_nts; 
  float param_ictype;
};

__constant__ GlobalConstants cuConstParams;

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
           float* x, float* y, int fnx, int fny){

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
    ps_old[C] = 5.625f - sqrtf( (xc-x[cent])*(xc-x[cent]) + yc*yc ) ;
    ps_new[C] = ps_old[C];
    U_old[C] = U_0;
    U_new[C] = U_0;
    ph_old = tanhf(ps_old[C]);
    ph_new = tanhf(ps_new[C]); 
  }
}

// anisotropy functions
__device__ float
atheta(float ux, float uz){
  
   float ux2 =  cosa*ux + sina*uz;
         ux2 = ux2*ux2;
   float uz2 = -sina*ux + cosa*uz;
         uz2 = uz2*uz2;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > eps){
         return a_s*( 1.0f + epsilon*(ux2*ux2 + uz2*uz2) / MAG_sq2);
   else {return 1.0f;}
}


__device__ float
aptheta(float ux, float uz){

   float uxr =  cosa*ux + sina*uz;
   float ux2 = uxr*uxr;
   float uzr = -sina*ux + cosa*uz;
   float uz2 = uzr*uzr;
   float MAG_sq = (ux2 + uz2);
   float MAG_sq2= MAG_sq*MAG_sq;
   if (MAG_sq > eps){
         return -a_12*uxr*uzr*(ux2 - uz2) / MAG_sq2;
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
       int R=C+1;
       int L=C-1;
       int T=C+fnx;
       int B=C-fnx;
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ps's are defined on cell centers
        psipjp=( ps[C] + ps[R] + ps[T] + ps[T+1] ) * 0.25f;
        psipjm=( ps[C] + ps[R] + ps[B] + ps[B+1] ) * 0.25f;
        psimjp=( ps[C] + ps[L] + ps[T-1] + ps[T] ) * 0.25f;
        psimjm=( ps[C] + ps[L] + ps[B-1] + ps[B] ) * 0.25f;

        phipjp=( ph[C] + ph[R] + ph[T] + ph[T+1] ) * 0.25f;
        phipjm=( ph[C] + ph[R] + ph[B] + ph[B+1] ) * 0.25f;
        phimjp=( ph[C] + ph[L] + ph[T-1] + ph[T] ) * 0.25f;
        phimjm=( ph[C] + ph[L] + ph[B-1] + ph[B] ) * 0.25f;
        
        // ============================
        // right edge flux
        // ============================
        psx = ps[R]-ps[C];
        psz = psipjp - psipjm;
        phx = ph[R]-ph[C];
        phz = phipjp - phipjm;

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        JR = A * ( A*psx - Ap*psz );
        
        // ============================
        // left edge flux
        // ============================
        psx = ps[C]-ps[L];
        psz = psimjp - psimjm;
        phx = ph[C]-ph[L];
        phz = phimjp - phimjm; 

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        JL = A * ( A*psx - Ap*psz );
        
        // ============================
        // top edge flux
        // ============================
        psx = psipjp - psimjp;
        psz = ps[T]-ps[C];
        phx = phipjp - phimjp;
        phz = ph[T]-ph[C];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        JT = A * ( A*psz + Ap*psx );

        // ============================
        // bottom edge flux
        // ============================
        psx = psipjm - psimjm;
        psz = ps[C]-ps[B];
        phx = phipjm - phimjm;
        phz = ph[C]-ph[B];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        JB = A * ( A*psz + Ap*psx );

         /*# =============================================================
        #
        # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
        #
        # =============================================================
        # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)*/
        phxn = ( ph[R] - ph[L] ) * 0.5f;
        phzn = ( ph[T] - ph[B] ) * 0.5f;
        psxn = ( ps[R] - ps[L] ) * 0.5f;
        pszn = ( ps[T] - ps[B] ) * 0.5f;

        A2 = atheta(phxn,phzn);
        A2 = A2*A2;
        gradps2 = (psxn)*(psxn) + (pszn)*(pszn);
        extra =  -sqrt2 * A2 * ph[C] * gradps2;

        /*# =============================================================
        #
        # 3. double well (transformed): sqrt2 * phi + nonlinear terms
        #
        # =============================================================*/

        Up = (y[j] - R_tilde * (nt*dt) )/lT_tilde;

        rhs_psi = ((JR-JL) + (JT-JB) + extra) * hi*hi + \
                   sqrt2*ph[C] - lamd*(1.0f-ph[C]*ph[C])*sqrt2*(U[C] + Up);

        /*# =============================================================
        #
        # 4. dpsi/dt term
        #
        # =============================================================*/
        tp = (1.0f-(1.0f-k)*Up);
        if (tp >= k){tau_psi = tp*A2;}
               else {tau_psi = k*A2;}
        
        dpsi[C] = rhs_psi / tau_psi; 
        
        ps_new[C] = ps[C] +  dt * dpsi[C];
        ph_new[C] = tanhf(ps_new[C]/sqrt2);
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

        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ps's are defined on cell centers
        phipjp=( ph[C] + ph[R] + ph[T] + ph[T+1] ) * 0.25f;
        phipjm=( ph[C] + ph[R] + ph[B] + ph[B+1] ) * 0.25f;
        phimjp=( ph[C] + ph[L] + ph[T-1] + ph[T] ) * 0.25f;
        phimjm=( ph[C] + ph[L] + ph[B-1] + ph[B] ) * 0.25f;

        jat    = 0.5f*(1.0f+(1.0f-k)*U[C])*(1.0f-ph[C]*ph[C])*dpsi[C];
        /*# ============================
        # right edge flux (i+1/2, j)
        # ============================*/
        phx = ph[R]-ph[C];
        phz = phipjp - phipjm;
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else { nx = 0.0f;}
        
        jat_ip = 0.5f*(1.0f+(1.0f-k)*U[R])*(1.0f-ph[R]*ph[R])*dpsi[R];	
        UR = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[R])*(U[R]-U[C]) + 0.5f*(jat + jat_ip)*nx;
    	 
    	 
        /* ============================
        # left edge flux (i-1/2, j)
        # ============================*/
        phx = ph[C]-ph[L];
        phz = phimjp - phimjm;
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else { nx = 0.0f;}
        
        jat_im = 0.5f*(1.0f+(1.0f-k)*U[L])*(1.0f-ph[L]*ph[L])*dpsi[L];
        UL = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[L])*(U[C]-U[L]) + 0.5f*(jat + jat_im)*nx;
    	 
    	 
        /*# ============================
        # top edge flux (i, j+1/2)
        # ============================*/     
        phx = phipjp - phimjp;
        phz = ph[T]-ph[C];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else { nz = 0.0f;}    	
  
        jat_jp = 0.5f*(1.0f+(1.0f-k)*U[T])*(1.0f-ph[T]*ph[T])*dpsi[T];      
        
        UT = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[T])*(U[T]-U[C]) + 0.5f*(jat + jat_jp)*nz;
    	 
    	 
        /*# ============================
        # bottom edge flux (i, j-1/2)
        # ============================*/  
        phx = phipjm - phimjm;
        phz = ph[C]-ph[B];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else { nz = 0.0f;} 

        jat_jm = 0.5f*(1.0f+(1.0f-k)*U[B])*(1.0f-ph[B]*ph[B])*dpsi[B];              
        UB = hi*Dl_tilde*0.5f*(2.0f - ph[C] - ph[B])*(U[C]-U[B]) + 0.5f*(jat + jat_jm)*nz;
        
        rhs_U = ( (UR-UL) + (UT-UB) ) * hi + sqrt2 * jat;
        tau_U = (1.0f+k) - (1.0f-k)*ph[C];

        U_new[C] = U[C] + dt * ( rhs_U / tau_U );

       }
}

void setup(float param_nx, float param_ny, float param_G, float param_R, float param_delta,
  float param_k, float param_c_infm, float param_Dl, float param_d0, float param_W0, 
  float param_lT, float param_lamd, float param_tau0, float param_c_infty, float param_R_tilde,
  float param_Dl_tilde, float param_lT_tilde, float param_eps, float param_alpha0, float param_dx, 
  float param_dt, float param_asp_ratio, float param_lxd, float param_lyd, float param_Mt,
  float param_eta, float param_U0, float param_nts, float param_ictype,
  float* x, float* y, float* phi, float* psi,float* U){
  // we should have already pass all the data structure in by this time
  // move those data onto device

  float* x_device = NULL;
  float* y_device = NULL;

  float* psi_old = NULL;
  float* psi_new = NULL;
  float* U_old = NULL;
  float* U_new = NULL;
  float* phi_old = NULL;
  float* phi_new = NULL;

  // allocate x, y, phi, psi, U related params
  int length_x=(int)param_nx+1;
  int length_y=(int)param_ny+1;
  cudaMalloc(&x_device, sizeof(float) * length_x);
  cudaMalloc(&y_device, sizeof(float) * length_y);

  int length = ((int)param_nx+3) * ((int)param_ny+3);
  cudaMalloc(&phi_old,  sizeof(float) * length);
  cudaMalloc(&psi_old,  sizeof(float) * length);
  cudaMalloc(&U_old,    sizeof(float) * length);
  cudaMalloc(&phi_new,  sizeof(float) * length);
  cudaMalloc(&psi_new,  sizeof(float) * length);
  cudaMalloc(&U_new,    sizeof(float) * length);

  cudaMemcpy(x_device, x, sizeof(float) * length_x, cudaMemcpyHostToDevice);
  cudaMemcpy(y_device, y, sizeof(float) * length_y, cudaMemcpyHostToDevice);
  cudaMemcpy(psi_old, psi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(phi_old, phi, sizeof(float) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(U_old, U, sizeof(float) * length, cudaMemcpyHostToDevice);

  // pass all the read-only params into global constant
  GlobalConstants params;
  params.param_nx = param_nx;
  params.param_ny = param_ny;
  params.param_G  = param_G;
  params.param_R  = param_R;
  params.param_delta  = param_delta;
  params.param_k  = param_k;
  params.param_c_infm = param_c_infm;
  params.param_Dl = param_Dl;
  params.param_d0  = param_d0;
  params.param_W0  = param_W0;
  params.param_lT  = param_lT;
  params.param_lamd  = param_lamd;
  params.param_tau0 = param_tau0;
  params.param_c_infty = param_c_infty;
  params.param_R_tilde  = param_R_tilde;
  params.param_Dl_tilde  = param_Dl_tilde;
  params.param_lT_tilde  = param_lT_tilde;
  params.param_eps  = param_eps;
  params.param_alpha0 = param_alpha0;
  params.param_dx = param_dx;
  params.param_dt  = param_dt;
  params.param_asp_ratio  = param_asp_ratio;
  params.param_lxd  = param_lxd;
  params.param_lyd  = param_lyd;
  params.param_Mt  = param_Mt;
  params.param_eta  = param_eta;
  params.param_U0  = param_U0;
  params.param_nts = param_nts;
  params.param_ictype = param_ictype;

  cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

}


void time_marching(){

   // initialize or load

   int blocksize_1d = 256;
   int blocksize_2d = 512;
   int num_block_2d = (fnx*fny+blocksize_2d-1)/blocksize_2d;
   int num_block_1d = (fnx+fny+blocksize_1d-1)/blocksize_1d;

   for (int kt=0; kt<Mt/2; kt++){

     rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, y, dpsi, fnx, fny, 2*kt ); 
     set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, fnx, fny);
     rhs_U<<< num_block_2d, blocksize_2d >>>(U_old, U_new, phi_new, dpsi);


     rhs_psi<<< num_block_2d, blocksize_2d >>>(psi_new, phi_new, U_new, psi_old, phi_old, y, dpsi, fnx, fny, 2*kt+1 ); 
     set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, fnx, fny);
     rhs_U<<< num_block_2d, blocksize_2d >>>(U_new, U_old, phi_old, dpsi);


   }

   
 
}




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
