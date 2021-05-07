#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include "CycleTimer.h"

using namespace std;

// this is dependent on the time tiling and grid size of one thread block
// we first finish a non-time tiling version
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define HALO_LEFT 1
#define HALO_RIGHT 1
#define HALO_TOP 1
#define HALO_BOTTOM 1

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
// TODO: this function is doing what, we can definetly merge this into kenrel right?
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
  // obtain i and j(2D position)
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

// psi & phi equation: two dimensions
__global__ void
rhs_psi_shared_mem(float* ps, float* ph, float* U, float* ps_new, float* ph_new, \
        float* y, float* dpsi, int fnx, int fny, int nt, int num_block_x, int num_block_y){
  // each CUDA theard block compute one grid(32*32)

  // memory access is from global and also not continous which cannot reach the max bandwidth(memory colaseing)
  // add a shared memory version to store the neighbours data: ps and ph
  // clare shared memory for time tiling
  // we have extra (nx+2)(ny+2) size of space to load
  int halo_left   = 1;
  int halo_right  = 1;
  int halo_top    = 1;
  int halo_bottom = 1;
  int real_block_x = BLOCK_DIM_X - halo_left - halo_right;
  int real_block_y = BLOCK_DIM_Y - halo_top  - halo_bottom;

  // load (32+2)*(32+2) daat from mem
  __shared__ float ps_shared[(BLOCK_DIM_Y)*(BLOCK_DIM_X)];
  __shared__ float ph_shared[(BLOCK_DIM_Y)*(BLOCK_DIM_X)];
  __shared__ float U_shared[(BLOCK_DIM_Y)*(BLOCK_DIM_X)];
  
  // each thread --> one data in one enlarged block
  int local_id = threadIdx.x; //local id -> sava to shared mem
  int local_id_x = local_id % BLOCK_DIM_X;
  int local_id_y = local_id / BLOCK_DIM_X;
  
  // obtain the block id in shrinked region
  int block_id = blockIdx.x; // 0~num_block_x*num_block_y 
  int block_id_x = block_id % num_block_x;
  int block_id_y = block_id / num_block_x;
  
  // this is the addr in inner region without considering the BC
  int block_addr_x = block_id_x * real_block_x;
  int block_addr_y = block_id_y * real_block_y;

  // find the addr of data in global memory
  // add 1 as we counter the block in inner region; - halo_left as we have halo region in this block
  int data_addr_x = block_addr_x + 1 - halo_left + local_id_x;
  int data_addr_y = block_addr_y + 1 - halo_bottom + local_id_y;

  int C= data_addr_y * fnx + data_addr_x; // place in global memory
  int j=C/fnx; 
  int i=C-j*fnx;

  // update data
  ps_shared[local_id] = ps[C];
  ph_shared[local_id] = ph[C];
  U_shared[local_id]  = U[C];

  __syncthreads();
  // if (C==1001){
  //   printf("check data 1: %f\n", ps[C]);
  // }

  // if (local_id == 0) printf("check data %f", ps_shared[local_id]);
  
  // compute based on the shared memory, skip if we are at the boundary
  int place = local_id_y * BLOCK_DIM_X + local_id_x;
  // if the points are at boundary, return
  // two levels of retunr: global and local region
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
      if ((local_id_x>0)&& (local_id_x<BLOCK_DIM_X-1) && (local_id_y>0) && (local_id_y<BLOCK_DIM_Y-1)) {
       // find the indices of the 8 neighbors for center
      //  if (C==1000){printf("find");}
       int R=place+1;
       int L=place-1;
       int T=place+BLOCK_DIM_X;
       int B=place-BLOCK_DIM_X;
       
      // //  preload data
      //  float ps_shared_c = ps_shared[place];
      //  float ps_shared_r = ps_shared[R];
      //  float ps_shared_l = ps_shared[L];
      //  float ps_shared_top = ps_shared[T];
      //  float ps_shared_top_l = ps_shared[T-1];
      //  float ps_shared_top_r = ps_shared[T+1];
      //  float ps_shared_b_r = ps_shared[B-1];
      //  float ps_shared_b_l = ps_shared[B+1];
      //  float ps_shared_b = ps_shared[B];

      //  float ph_shared_c = ph_shared[place];
      //  float ph_shared_r = ph_shared[R];
      //  float ph_shared_l = ph_shared[L];
      //  float ph_shared_top = ph_shared[T];
      //  float ph_shared_top_l = ph_shared[T-1];
      //  float ph_shared_top_r = ph_shared[T+1];
      //  float ph_shared_b_r = ph_shared[B-1];
      //  float ph_shared_b_l = ph_shared[B+1];
      //  float ph_shared_b = ph_shared[B];
      //  if (C==1001){
      //    printf("detailed check of neighbours\n");
      //    printf("R: %d ; L:%d ; T: %d ; B: %d \n", R, L, T, B);
      //    printf("R: %f ; L:%f ; T: %f ; B: %f \n", ps_shared[R], ps_shared[L], ps_shared[T], ps_shared[B]);
      //    printf("R: %f ; L:%f ; T: %f ; B: %f \n", ps_shared[R], ps_shared[L], ps_shared[T+1], ps_shared[B]);
      //  }
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ps's are defined on cell centers
        float psipjp=( ps_shared[place] + ps_shared[R] + ps_shared[T] + ps_shared[T+1] ) * 0.25f;
        float psipjm=( ps_shared[place] + ps_shared[R] + ps_shared[B] + ps_shared[B+1] ) * 0.25f;
        float psimjp=( ps_shared[place] + ps_shared[L] + ps_shared[T-1] + ps_shared[T] ) * 0.25f;
        float psimjm=( ps_shared[place] + ps_shared[L] + ps_shared[B-1] + ps_shared[B] ) * 0.25f;

        float phipjp=( ph_shared[place] + ph_shared[R] + ph_shared[T] + ph_shared[T+1] ) * 0.25f;
        float phipjm=( ph_shared[place] + ph_shared[R] + ph_shared[B] + ph_shared[B+1] ) * 0.25f;
        float phimjp=( ph_shared[place] + ph_shared[L] + ph_shared[T-1] + ph_shared[T] ) * 0.25f;
        float phimjm=( ph_shared[place] + ph_shared[L] + ph_shared[B-1] + ph_shared[B] ) * 0.25f;
        
        // if (C==1001){
        //   printf("detailed check of neighbours 2\n");
        //   printf("R: %f ; L:%f ; T: %f ; B: %f \n", psipjp, psipjm, psimjp, psimjm);
        // }

        // ============================
        // right edge flux
        // ============================
        float psx = ps_shared[R]-ps_shared[place];
        float psz = psipjp - psipjm;
        float phx = ph_shared[R]-ph_shared[place];
        float phz = phipjp - phipjm;

        float A  = atheta( phx,phz);
        float Ap = aptheta(phx,phz);
        float JR = A * ( A*psx - Ap*psz );
        
        // ============================
        // left edge flux
        // ============================
        psx = ps_shared[place]-ps_shared[L];
        psz = psimjp - psimjm;
        phx = ph_shared[place]-ph_shared[L];
        phz = phimjp - phimjm; 

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JL = A * ( A*psx - Ap*psz );

        // ============================
        // top edge flux
        // ============================
        psx = psipjp - psimjp;
        psz = ps_shared[T]-ps_shared[place];
        phx = phipjp - phimjp;
        phz = ph_shared[T]-ph_shared[place];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JT = A * ( A*psz + Ap*psx );

        // ============================
        // bottom edge flux
        // ============================
        psx = psipjm - psimjm;
        psz = ps_shared[place]-ps_shared[B];
        phx = phipjm - phimjm;
        phz = ph_shared[place]-ph_shared[B];

        A  = atheta( phx,phz);
        Ap = aptheta(phx,phz);
        float JB = A * ( A*psz + Ap*psx );

        // if (C==1001){
        //   printf("detailed check of neighbours 3\n");
        //   printf("R: %f ; L:%f ; T: %f ; B: %f \n", JR, JL, JT, JB);
        // }
         /*# =============================================================
        #
        # 2. EXTRA TERM: sqrt2 * atheta**2 * phi * |grad psi|^2
        #
        # =============================================================
        # d(phi)/dx  d(psi)/dx d(phi)/dz  d(psi)/dz at nodes (i,j)*/
        float phxn = ( ph_shared[R] - ph_shared[L] ) * 0.5f;
        float phzn = ( ph_shared[T] - ph_shared[B] ) * 0.5f;
        float psxn = ( ps_shared[R] - ps_shared[L] ) * 0.5f;
        float pszn = ( ps_shared[T] - ps_shared[B] ) * 0.5f;

        float A2 = atheta(phxn,phzn);
        A2 = A2*A2;
        float gradps2 = (psxn)*(psxn) + (pszn)*(pszn);
        float extra =  -cP.sqrt2 * A2 * ph_shared[place] * gradps2;

        /*# =============================================================
        #
        # 3. double well (transformed): sqrt2 * phi + nonlinear terms
        #
        # =============================================================*/

        float Up = (y[j]/cP.W0 - cP.R_tilde * (nt*cP.dt) )/cP.lT_tilde;

        float rhs_psi = ((JR-JL) + (JT-JB) + extra) * cP.hi*cP.hi + \
                   cP.sqrt2*ph_shared[place] - cP.lamd*(1.0f-ph_shared[place]*ph_shared[place])*cP.sqrt2*(U_shared[place] + Up);

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
        
        ps_new[C] = ps_shared[place] +  cP.dt * dpsi[C];
        ph_new[C] = tanhf(ps_new[C]/cP.sqrt2);
        // if (C==1000){printf("%f ",ph_new[C]);}
        // if (C == 1000) printf("check data %f\n", ps_shared[local_id]);
        // if (C == 137) {
        //   printf("check data ps: %f and ph: %f and dpsi: %f and U: %f\n", ps_new[C], ph_new[C], dpsi[C], U[C]);
        //   // printf("block id %d ; local_id_x %d; local_id_y %d\n", block_id, local_id_x, local_id_y);
        //   // printf("block id %d ; data_addr_x %d; data_addr_y %d\n", block_id, data_addr_x, data_addr_y);
        // }
         }
  }
  // __syncthreads();
} 

// U equation
__global__ void
rhs_U_shared_mem(float* U, float* U_new, float* ph, float* dpsi, int fnx, int fny, int num_block_x, int num_block_y){
  // __shared__ float ps_shared[(BLOCK_DIM_Y)*(BLOCK_DIM_X)];
  __shared__ float ph_shared[(BLOCK_DIM_Y)*(BLOCK_DIM_X)];
  __shared__ float U_shared[(BLOCK_DIM_Y)*(BLOCK_DIM_X)];
  __shared__ float dpsi_shared[(BLOCK_DIM_Y)*(BLOCK_DIM_X)];

  int halo_left   = 1;
  int halo_right  = 1;
  int halo_top    = 1;
  int halo_bottom = 1;
  int real_block_x = BLOCK_DIM_X - halo_left - halo_right;
  int real_block_y = BLOCK_DIM_Y - halo_top  - halo_bottom;

  // each thread --> one data in one enlarged block
  int local_id = threadIdx.x; //local id -> sava to shared mem
  int local_id_x = local_id % BLOCK_DIM_X;
  int local_id_y = local_id / BLOCK_DIM_X;
  
  // obtain the block id in shrinked region
  int block_id = blockIdx.x; // 0~num_block_x*num_block_y 
  int block_id_x = block_id % num_block_x;
  int block_id_y = block_id / num_block_x;
  
  // this is the addr in inner region without considering the BC
  int block_addr_x = block_id_x * real_block_x;
  int block_addr_y = block_id_y * real_block_y;

  // find the addr of data in global memory
  // add 1 as we counter the block in inner region; - halo_left as we have halo region in this block
  int data_addr_x = block_addr_x + 1 + local_id_x - halo_left;
  int data_addr_y = block_addr_y + 1 - halo_bottom + local_id_y;

  int C= data_addr_y * fnx + data_addr_x; // place in global memory
  // int j=C/fnx; 
  // int i=C-j*fnx;
  int j = data_addr_y;
  int i = data_addr_x;

  // update data
  // ps_shared[local_id] = ps[C];
  ph_shared[local_id] = ph[C];
  U_shared[local_id]  = U[C];
  dpsi_shared[local_id]  = dpsi[C];
  
  __syncthreads();
  // if (C==1001){
  //   printf("check pre-loaded data\n");
  //   printf("ph: %f ; u:%f ; dpsi: %f\n", ph[C], U[C], dpsi[C]);
  // }
  if ((i > fnx - 1) ||(i > fny - 1)) {return;}

  // if (C==137){
  //   printf("check pre-loaded data\n");
  //   printf("local id: %d and block id: %d\n", local_id, block_id);
  //   // printf("data_addr_x: %d and data_addr_y: %d\n", data_addr_x, data_addr_y);
  //   // printf("i: %d and j: %d\n", i, j);
  //   printf("ph: %f ; u:%f ; dpsi: %f\n", ph[C], U[C], dpsi[C]);
  //   printf("ph: %f ; u:%f ; dpsi: %f\n", ph_shared[6], U_shared[6], dpsi_shared[6]);
  //   printf("ph: %f ; u:%f ; dpsi: %f\n", ph_shared[0], U_shared[0], dpsi_shared[0]);
  //   printf("ph: %f ; u:%f ; dpsi: %f\n", ph_shared[real_block_x+halo_left+halo_right-1], U_shared[real_block_x+halo_left+halo_right-1], dpsi_shared[real_block_x+halo_left+halo_right-1]);
  //   printf("ph: %f ; u:%f ; dpsi: %f\n", ph_shared[(real_block_x+halo_left+halo_right)*(real_block_y+halo_bottom)], U_shared[(real_block_x+halo_left+halo_right)*(real_block_y+halo_bottom)], dpsi_shared[(real_block_x+halo_left+halo_right)*(real_block_y+halo_bottom)]);
  //   printf("ph: %f ; u:%f ; dpsi: %f\n", ph_shared[(real_block_x+halo_left+halo_right)*(real_block_y+halo_bottom+halo_top) - 1], U_shared[(real_block_x+halo_left+halo_right)*(real_block_y+halo_bottom+halo_top) - 1], dpsi_shared[(real_block_x+halo_left+halo_right)*(real_block_y+halo_bottom+halo_top) - 1]);
  // }

  int place = local_id_y * BLOCK_DIM_X + local_id_x;
  // if the points are at boundary, return
  if ( (i>0) && (i<fnx-1) && (j>0) && (j<fny-1) ) {
    if ((local_id_x>0)&& (local_id_x<BLOCK_DIM_X-1) && (local_id_y>0) && (local_id_y<BLOCK_DIM_Y-1)) {
        // find the indices of the 8 neighbors for center
        int R=place+1;
        int L=place-1;
        int T=place+BLOCK_DIM_X;
        int B=place-BLOCK_DIM_X;
        float hi = cP.hi;
        float Dl_tilde = cP.Dl_tilde;
        float k = cP.k;
        float nx, nz;
        float eps = cP.eps;
        // =============================================================
        // 1. ANISOTROPIC DIFFUSION
        // =============================================================

        // these ph's are defined on cell centers
        float phipjp=( ph_shared[place] + ph_shared[R] + ph_shared[T] + ph_shared[T+1] ) * 0.25f;
        float phipjm=( ph_shared[place] + ph_shared[R] + ph_shared[B] + ph_shared[B+1] ) * 0.25f;
        float phimjp=( ph_shared[place] + ph_shared[L] + ph_shared[T-1] + ph_shared[T] ) * 0.25f;
        float phimjm=( ph_shared[place] + ph_shared[L] + ph_shared[B-1] + ph_shared[B] ) * 0.25f;

        // if (C==137){
        //   printf("detailed check of neighbours 3\n");
        //   printf("R: %f ; L:%f ; T: %f ; B: %f \n", phipjp, phipjm, phimjp, phimjm);
        //   printf("R: %f ; L:%f ; T: %f ; B: %f \n", U_shared[R], U_shared[L], U_shared[T], U_shared[B]);
        //   printf("R: %f ; L:%f ; T: %f ; B: %f \n", dpsi_shared[R], dpsi_shared[L], dpsi_shared[T], dpsi_shared[B]);
        // }
        float jat    = 0.5f*(1.0f+(1.0f-k)*U_shared[place])*(1.0f-ph_shared[place]*ph_shared[place])*dpsi_shared[place];
        /*# ============================
        # right edge flux (i+1/2, j)
        # ============================*/
        float phx = ph_shared[R]-ph_shared[place];
        float phz = phipjp - phipjm;
        float phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else {nx = 0.0f;}
        
        float jat_ip = 0.5f*(1.0f+(1.0f-k)*U_shared[R])*(1.0f-ph_shared[R]*ph_shared[R])*dpsi_shared[R];	
        float UR = hi*Dl_tilde*0.5f*(2.0f - ph_shared[place] - ph_shared[R])*(U_shared[R]-U_shared[place]) + 0.5f*(jat + jat_ip)*nx;
    	 
        /* ============================
        # left edge flux (i-1/2, j)
        # ============================*/
        phx = ph_shared[place]-ph_shared[L];
        phz = phimjp - phimjm;
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nx = phx / sqrtf(phn2);}
                   else {nx = 0.0f;}
        
        float jat_im = 0.5f*(1.0f+(1.0f-k)*U_shared[L])*(1.0f-ph_shared[L]*ph_shared[L])*dpsi_shared[L];
        float UL = hi*Dl_tilde*0.5f*(2.0f - ph_shared[place] - ph_shared[L])*(U_shared[place]-U_shared[L]) + 0.5f*(jat + jat_im)*nx;
    	 
    	 
        /*# ============================
        # top edge flux (i, j+1/2)
        # ============================*/     
        phx = phipjp - phimjp;
        phz = ph_shared[T]-ph_shared[place];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else {nz = 0.0f;}    	
  
        float jat_jp = 0.5f*(1.0f+(1.0f-k)*U_shared[T])*(1.0f-ph_shared[T]*ph_shared[T])*dpsi_shared[T];      
        
        float UT = hi*Dl_tilde*0.5f*(2.0f - ph_shared[place] - ph_shared[T])*(U_shared[T]-U_shared[place]) + 0.5f*(jat + jat_jp)*nz;
    	 
    	 
        /*# ============================
        # bottom edge flux (i, j-1/2)
        # ============================*/  
        phx = phipjm - phimjm;
        phz = ph_shared[place]-ph_shared[B];
        phn2 = phx*phx + phz*phz;
        if (phn2 > eps) {nz = phz / sqrtf(phn2);}
                   else {nz = 0.0f;} 

        float jat_jm = 0.5f*(1.0f+(1.0f-k)*U_shared[B])*(1.0f-ph_shared[B]*ph_shared[B])*dpsi_shared[B];              
        float UB = hi*Dl_tilde*0.5f*(2.0f - ph_shared[place] - ph_shared[B])*(U_shared[place]-U_shared[B]) + 0.5f*(jat + jat_jm)*nz;
        
        // if (C==137){
        //   printf("detailed check of neighbours 4\n");
        //   printf("UR: %f ; UL:%f ; UT: %f ; UB: %f \n", UR, UL, UT, UB);
        // }

        float rhs_U = ( (UR-UL) + (UT-UB) ) * hi + cP.sqrt2 * jat;
        float tau_U = (1.0f+cP.k) - (1.0f-cP.k)*ph_shared[place];

        U_new[C] = U_shared[place] + cP.dt * ( rhs_U / tau_U );
        // if (C==137){
        //   printf("detailed check of neighbours 3\n");
        //   printf("U: %f \n", U_new[C]);
        // }
       }
  }
}


// U equation
__global__ void
rhs_U_ori(float* U, float* U_new, float* ph, float* dpsi, int fnx, int fny ){

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

// Host codes for PF computing
void setup(GlobalConstants params, int fnx, int fny, float* x, float* y, float* phi, float* psi,float* U){
  // we should have already pass all the data structure in by this time
  // move those data onto device
  printCudaInfo();
  float* x_device;// = NULL;
  float* y_device;// = NULL;
  // store two for swap behavior
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

  // set initial params
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
   printf("nx: %d and ny: %d\n", fnx, fny);
   printf("block size %d, # blocks %d\n", blocksize_2d, num_block_2d); 
   initialize<<< num_block_2d, blocksize_2d >>>(psi_old, phi_old, U_old, psi_new, phi_new, U_new, x_device, y_device, fnx, fny);
   set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_new, dpsi, fnx, fny);
   set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_old, dpsi, fnx, fny);
   cudaDeviceSynchronize();
   double startTime = CycleTimer::currentSeconds();
   
  // change the 2d block due to we donn't want to include halo region
    int real_per_block_x = BLOCK_DIM_X - HALO_LEFT - HALO_RIGHT;
    int real_per_block_y = BLOCK_DIM_Y - HALO_TOP - HALO_BOTTOM;
    int num_block_x = (fnx - 2 + real_per_block_x - 1) / real_per_block_x;
    int num_block_y = (fny - 2 + real_per_block_y - 1) / real_per_block_y;
    printf("block_x: %d and block_y: %d\n", real_per_block_x, real_per_block_y);
    printf("block_x: %d and block_y: %d\n", num_block_x, num_block_y);
    int num_block_2d_s = num_block_x * num_block_y; //each one take one block with (32-2)+ (32-2) ture block within (fnx-2), (fny-2)
    int blocksize_2d_s = BLOCK_DIM_X * BLOCK_DIM_Y; // 32*32: as we have to write 32*32 data region into shared memory
   for (int kt=0; kt<params.Mt/2; kt++){
   //  printf("time step %d\n",kt);
     rhs_psi_shared_mem<<< num_block_2d_s, blocksize_2d_s >>>(psi_old, phi_old, U_old, psi_new, phi_new, y_device, dpsi, fnx, fny, 2*kt, num_block_x, num_block_y);
    //  cudaDeviceSynchronize();
     set_BC<<< num_block_1d, blocksize_1d >>>(psi_new, phi_new, U_old, dpsi, fnx, fny);
    //  cudaDeviceSynchronize();
     rhs_U_shared_mem<<< num_block_2d_s, blocksize_2d_s >>>(U_old, U_new, phi_new, dpsi, fnx, fny, num_block_x, num_block_y);
    //  cudaDeviceSynchronize();

     rhs_psi_shared_mem<<< num_block_2d_s, blocksize_2d_s >>>(psi_new, phi_new, U_new, psi_old, phi_old, y_device, dpsi, fnx, fny, 2*kt+1, num_block_x, num_block_y);
    //  cudaDeviceSynchronize();
     set_BC<<< num_block_1d, blocksize_1d >>>(psi_old, phi_old, U_new, dpsi, fnx, fny);
    //  cudaDeviceSynchronize();
     rhs_U_shared_mem<<< num_block_2d_s, blocksize_2d_s >>>(U_new, U_old, phi_old, dpsi, fnx, fny, num_block_x, num_block_y);
    //  cudaDeviceSynchronize();
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
