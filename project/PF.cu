#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

using namespace std;

extern float toBW(int bytes, float sec);


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
rhs_U(float* ph, float* U, float* U_new, float* dpsi, int fnx, int fny ){

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

void setup()


void time_marching(){

   // initialize or load

   int blocksize = 256;
   int num_block_2d = (fnx*fny+blocksize-1)/blocksize;
   int num_block_1d = (fnx+fny+blocksize-1)/blocksize;

   for (int kt=0; kt<Mt; kt++){

     rhs_psi<<<>>>(psi_old, phi_old, U_old, psi_new, phi_new, y, dpsi, fnx, fny, 2*kt ) 
     set_BC(float* ps, float* ph, float* U, float* dpsi, infnx, fny)

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