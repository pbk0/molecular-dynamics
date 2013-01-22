//################################################################################
//#
//# Author: Praveen Baburao Kulkarni praveen@spikingneurons.com
//#
//################################################################################
//#
//# 
//#
//################################################################################



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////  Function translates cublas error codes to simple string message
////////////////////////////////////////////////////////////////////////////////
static const char*
cublasErrStr(int err_code)
{
  switch(err_code)
  {
    case CUBLAS_STATUS_SUCCESS         :
      return "CUBLAS_STATUS_SUCCESS"          ;
    case CUBLAS_STATUS_NOT_INITIALIZED :
      return "CUBLAS_STATUS_NOT_INITIALIZED"  ;
    case CUBLAS_STATUS_ALLOC_FAILED    :
      return "CUBLAS_STATUS_ALLOC_FAILED"     ;
    case CUBLAS_STATUS_INVALID_VALUE   :
      return "CUBLAS_STATUS_INVALID_VALUE"    ;
    case CUBLAS_STATUS_ARCH_MISMATCH   :
      return "CUBLAS_STATUS_ARCH_MISMATCH"    ;
    case CUBLAS_STATUS_MAPPING_ERROR   :
      return "CUBLAS_STATUS_MAPPING_ERROR"    ;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED" ;
    case CUBLAS_STATUS_INTERNAL_ERROR  :
      return "CUBLAS_STATUS_INTERNAL_ERROR"   ;
    default                            :
      return "Unknown CUBLAS ERROR"           ;
  } ;
} ;
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////GPU VARIABLES AND MEM COPY METHODS
////////////////////////////////////////////////////////////////////////////////

static int BLOCK_SIZE=-12;
double *    acc_gpu;
double *    force_gpu;
double *    pos_gpu;
double *    vel_gpu;
double *    parpot_gpu;
double *    parforce_gpu;
double *    velnew_gpu;
double *    accnew_gpu;

//////////////////////////
void allocateArrayOnGPU_accnew(int size)
{
    cudaError_t err = cudaMalloc ((void**)(&accnew_gpu), size*sizeof(double));
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: allocateArrayOnGPU_accnew FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void copyArrayOnGPU_accnew(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(accnew_gpu, hostPtr, size*sizeof(double), cudaMemcpyHostToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: copyArrayOnGPU_accnew FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void readArrayFromGPU_accnew(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(hostPtr, accnew_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: readArrayFromGPU_accnew FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void freeArrayFromGPU_accnew()
{
    cudaError_t err = cudaFree (accnew_gpu);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: freeArrayFromGPU_accnew FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};

//////////////////////////
void allocateArrayOnGPU_velnew(int size)
{
    cudaError_t err = cudaMalloc ((void**)(&velnew_gpu), size*sizeof(double));
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: allocateArrayOnGPU_velnew FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void copyArrayOnGPU_velnew(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(velnew_gpu, hostPtr, size*sizeof(double), cudaMemcpyHostToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: copyArrayOnGPU_velnew FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void readArrayFromGPU_velnew(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(hostPtr, velnew_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: readArrayFromGPU_velnew FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void freeArrayFromGPU_velnew()
{
    cudaError_t err = cudaFree (velnew_gpu);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: freeArrayFromGPU_velnew FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};

//////////////////////////
void allocateArrayOnGPU_parforce(int size)
{
    cudaError_t err = cudaMalloc ((void**)(&parforce_gpu), size*sizeof(double));
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: allocateArrayOnGPU_parforce FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void copyArrayOnGPU_parforce(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(parforce_gpu, hostPtr, size*sizeof(double), cudaMemcpyHostToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: copyArrayOnGPU_parforce FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void readArrayFromGPU_parforce(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(hostPtr, parforce_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: readArrayFromGPU_parforce FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void freeArrayFromGPU_parforce()
{
    cudaError_t err = cudaFree (parforce_gpu);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: freeArrayFromGPU_parforce FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};

//////////////////////////
void allocateArrayOnGPU_parpot(int size)
{
    cudaError_t err = cudaMalloc ((void**)(&parpot_gpu), size*sizeof(double));
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: allocateArrayOnGPU_parpot FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void copyArrayOnGPU_parpot(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(parpot_gpu, hostPtr, size*sizeof(double), cudaMemcpyHostToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: copyArrayOnGPU_parpot FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void readArrayFromGPU_parpot(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(hostPtr, parpot_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: readArrayFromGPU_parpot FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void freeArrayFromGPU_parpot()
{
    cudaError_t err = cudaFree (parpot_gpu);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: freeArrayFromGPU_parpot FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};

//////////////////////////
void allocateArrayOnGPU_acc(int size)
{
    cudaError_t err = cudaMalloc ((void**)(&acc_gpu), size*sizeof(double));
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: allocateArrayOnGPU_acc FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void copyArrayOnGPU_acc(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(acc_gpu, hostPtr, size*sizeof(double), cudaMemcpyHostToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: copyArrayOnGPU_acc FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void readArrayFromGPU_acc(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(hostPtr, acc_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: readArrayFromGPU_acc FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void freeArrayFromGPU_acc()
{
    cudaError_t err = cudaFree (acc_gpu);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: freeArrayFromGPU_acc FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};

//////////////////////////
void allocateArrayOnGPU_force(int size)
{
    cudaError_t err = cudaMalloc ((void**)(&force_gpu), size*sizeof(double));
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: allocateArrayOnGPU_force FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void copyArrayOnGPU_force(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(force_gpu, hostPtr, size*sizeof(double), cudaMemcpyHostToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: copyArrayOnGPU_force FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void readArrayFromGPU_force(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(hostPtr, force_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: readArrayFromGPU_force FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void freeArrayFromGPU_force()
{
    cudaError_t err = cudaFree (force_gpu);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: freeArrayFromGPU_force FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};

//////////////////////////
void allocateArrayOnGPU_pos(int size)
{
    cudaError_t err = cudaMalloc ((void**)(&pos_gpu), size*sizeof(double));
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: allocateArrayOnGPU_pos FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void copyArrayOnGPU_pos(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(pos_gpu, hostPtr, size*sizeof(double), cudaMemcpyHostToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: copyArrayOnGPU_pos FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void readArrayFromGPU_pos(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(hostPtr, pos_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: readArrayFromGPU_pos FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void freeArrayFromGPU_pos()
{
    cudaError_t err = cudaFree (pos_gpu);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: freeArrayFromGPU_pos FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};

//////////////////////////
void allocateArrayOnGPU_vel(int size)
{
    cudaError_t err = cudaMalloc ((void**)(&vel_gpu), size*sizeof(double));
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: allocateArrayOnGPU_vel FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void copyArrayOnGPU_vel(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(vel_gpu, hostPtr, size*sizeof(double), cudaMemcpyHostToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: copyArrayOnGPU_vel FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void readArrayFromGPU_vel(double* hostPtr,int size )
{
    cudaError_t err = cudaMemcpy(hostPtr, vel_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: readArrayFromGPU_vel FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};
void freeArrayFromGPU_vel()
{
    cudaError_t err = cudaFree (vel_gpu);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: freeArrayFromGPU_vel FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
};

/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

/////////////////////////////////////////////////////////////////////////// [--]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////  DUMMY KERNEL
////////////////////////////////////////////////////////////////////////////////
// Kernel code
static __global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
cuda_dummy_KERNEL()
{
}

/////////////////////////////////////////////////////////////////////////// [K1]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////  COMPUTE FORCE ON PARTICLE
////////////////////////////////////////////////////////////////////////////////
// Kernel code
static __global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
cuda_compute_forceonparticle_KERNEL(int np, int numberOfThreads, int currentMoleculeIndex, const double PI2, double * pos_gpu, double * parforce_gpu, double * parpot_gpu)
{
    double posx, posy, posz, dist, dist2;
    __shared__ double currentposx,currentposy,currentposz;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    
    if (idx >= numberOfThreads) return ;
    
    posx = pos_gpu[idx];
    posy = pos_gpu[idx+np];
    posz = pos_gpu[idx+np+np];
    currentposx = pos_gpu[currentMoleculeIndex];
    currentposy = pos_gpu[currentMoleculeIndex+np];
    currentposz = pos_gpu[currentMoleculeIndex+np+np];
    
    posx = currentposx - posx; 
    posy = currentposy - posy;
    posz = currentposz - posz;
    
    dist = posx*posx + posy*posy + posz*posz;
    dist = sqrt(dist);
    
    
    dist2 = (dist < PI2) ? dist : PI2;
    
    if(idx==currentMoleculeIndex){
    	posx=0.0;
    	posy=0.0;
    	posz=0.0;
    	dist=0.0;
    }else{
    	posx= - (posx * sin(2.0 * dist2) / dist);
    	posy= - (posy * sin(2.0 * dist2) / dist);
    	posz= - (posz * sin(2.0 * dist2) / dist);
    	dist= 0.5 * sin(dist2) * sin(dist2);
    }
    
    parforce_gpu[idx] = posx;
    parforce_gpu[idx+np] = posy;
    parforce_gpu[idx+np+np] = posz;
    parpot_gpu[idx] = dist;
} ;
// call to kernel code
void cuda_compute_forceonparticle(int nd, int np, int currentMoleculeIndex, const double PI2)
{
    int numberOfThreads = np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimGrid(numBlocks) ;
    dim3 dimBlock(BLOCK_SIZE) ;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    cuda_compute_forceonparticle_KERNEL<<<dimGrid,dimBlock>>>(np, numberOfThreads, currentMoleculeIndex, PI2, pos_gpu, parforce_gpu, parpot_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_compute_forceonparticle_KERNEL failed");
    return;
    #endif
    
    cuda_compute_forceonparticle_KERNEL<<<dimGrid,dimBlock>>>(np, numberOfThreads, currentMoleculeIndex, PI2, pos_gpu, parforce_gpu, parpot_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_compute_forceonparticle_KERNEL failed");
}

/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////  GENERAL REDUCTION KERNEL (uses shared memory)
////////////////////////////////////////////////////////////////////////////////

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

// This version uses sequential addressing -- no divergence or bank conflicts.
template <class T>
__global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
reduce_withsharedmemory_KERNEL(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////  GENERAL REDUCTION KERNEL (does not use shared memory)
////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
reduce_withoutsharedmemory_KERNEL(T *g_idata, T *g_odata, unsigned int numberOfThreads)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if (idx >= numberOfThreads) return ;
    g_odata[idx] = g_idata[idx] + g_idata[idx+numberOfThreads];
    if(idx==0)g_idata[numberOfThreads] = 0.0;
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

//////////////////////////////////////////////////////////////////////// [K2nsh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// CUMULATE parpot without using shared memory
////////////////////////////////////////////////////////////////////////////////
double cuda_cumulate_parpot_withoutsharedmemory(int nd, int np)
{
    if( np%2 != 0)
    {
        printf("\n\n>>>>> ERROR: Numpber of molecules (np) is %d. Please use even number of molecules.\n\n", np);
        exit(-1);
    };
    int numberOfThreads = np/2;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
    dim3 dimGrid;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    dimGrid = dim3(numBlocks, 1, 1);
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parpot_gpu, parpot_gpu, numberOfThreads);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parpot_withoutsharedmemory reduce_withoutsharedmemory_KERNEL failed while profiling");
    return 0.0;
    #endif
    
    while(numberOfThreads>256){  
        dimGrid = dim3(numBlocks, 1, 1);
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parpot_gpu, parpot_gpu, numberOfThreads);
        numberOfThreads = numberOfThreads/2 + (numberOfThreads % 2 == 0 ? 0 : 1) ;
        numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    }
    
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parpot_withoutsharedmemory reduce_withoutsharedmemory_KERNEL failed");
    
    numberOfThreads *=2;
    double* parpot = (double*) malloc(numberOfThreads * sizeof (double));    
    readArrayFromGPU_parpot(parpot,numberOfThreads);
    double sum = 0.0;
    for(int i=0; i<numberOfThreads; i++){
    	  sum += parpot[i];
    }
    
    return sum;
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

///////////////////////////////////////////////////////////////////////// [K2sh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////// CUMULATE parpot with help of generalized reduction kernel
////////////////////////////////////////////////////////////////////////////////
double cuda_cumulate_parpot_withsharedmemory(int nd, int np)
{
    int numberOfThreads = np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
    dim3 dimGrid;
    int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(double) : BLOCK_SIZE * sizeof(double);
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    dimGrid = dim3(numBlocks, 1, 1);
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parpot_gpu, parpot_gpu, numberOfThreads);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parpot_withsharedmemory reduce_withsharedmemory_KERNEL failed while profiling");
    return 0.0;
    #endif
    
    while(numberOfThreads>512){  
        dimGrid = dim3(numBlocks, 1, 1);
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parpot_gpu, parpot_gpu, numberOfThreads);
        numberOfThreads = numBlocks;
        numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    }
    
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parpot_withsharedmemory reduce_withsharedmemory_KERNEL failed");
    
    double* parpot = (double*) malloc(numberOfThreads * sizeof (double));    
    readArrayFromGPU_parpot(parpot,numberOfThreads );
    double sum = 0.0;
    for(int i=0; i<numberOfThreads; i++){
    	  sum += parpot[i];
    }
    
    return sum;
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

//////////////////////////////////////////////////////////////////////// [K3nsh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// CUMULATE parforce without help of shared memory
////////////////////////////////////////////////////////////////////////////////
void cuda_cumulate_parforce_withoutsharedmemory(int nd, int np, int currentMoleculeIndex)
{    
    if( np%2 != 0)
    {
        printf("\n\n>>>>> ERROR: Numpber of molecules (np) is %d. Please use even number of molecules.\n\n", np);
        exit(-1);
    };
    int numberOfThreads = np/2;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
    dim3 dimGrid;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    dimGrid = dim3(numBlocks, 1, 1); 
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu, parforce_gpu, numberOfThreads);
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu+(np), parforce_gpu+(np), numberOfThreads);
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu+(np*2), parforce_gpu+(np*2), numberOfThreads); 
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parforce_withoutsharedmemory reduce_withoutsharedmemory_KERNEL failed while profiling"); 
    return;
    #endif
    
    while(numberOfThreads>1){    
        dimGrid = dim3(numBlocks, 1, 1); 
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu, parforce_gpu, numberOfThreads);
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu+(np), parforce_gpu+(np), numberOfThreads);
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu+(np*2), parforce_gpu+(np*2), numberOfThreads);  
        numberOfThreads = numberOfThreads/2 + (numberOfThreads % 2 == 0 ? 0 : 1) ;
        numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;  	
    }
    
    dimGrid = dim3(numBlocks, 1, 1); 
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu, force_gpu+currentMoleculeIndex, numberOfThreads);
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu+(np), force_gpu+(np+currentMoleculeIndex), numberOfThreads);
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu+(np*2), force_gpu+(np+np+currentMoleculeIndex), numberOfThreads);  
    
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parforce_withoutsharedmemory reduce_withoutsharedmemory_KERNEL failed"); 
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

///////////////////////////////////////////////////////////////////////// [K3sh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// CUMULATE parforce with help of shared memory
////////////////////////////////////////////////////////////////////////////////
void cuda_cumulate_parforce_withsharedmemory(int nd, int np, int currentMoleculeIndex)
{    
    int numberOfThreads = np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
    dim3 dimGrid;
    int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(double) : BLOCK_SIZE * sizeof(double); 
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    dimGrid = dim3(numBlocks, 1, 1); 
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu, parforce_gpu, numberOfThreads);
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu+(np), parforce_gpu+(np), numberOfThreads);
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu+(np*2), parforce_gpu+(np*2), numberOfThreads);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parforce_withsharedmemory reduce_withsharedmemory_KERNEL failed while profiling"); 
    return;
    #endif
    
    while(numBlocks>1){    
        dimGrid = dim3(numBlocks, 1, 1); 
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu, parforce_gpu, numberOfThreads);
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu+(np), parforce_gpu+(np), numberOfThreads);
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu+(np*2), parforce_gpu+(np*2), numberOfThreads);  
        numberOfThreads = numBlocks;
        numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;  	
    }
    
    dimGrid = dim3(numBlocks, 1, 1); 
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu, force_gpu+currentMoleculeIndex, numberOfThreads);
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu+(np), force_gpu+(np+currentMoleculeIndex), numberOfThreads);
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu+(np*2), force_gpu+(np+np+currentMoleculeIndex), numberOfThreads);  
    
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parforce_withsharedmemory reduce_withsharedmemory_KERNEL failed"); 
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

//////////////////////////////////////////////////////////////////////// [K4nsh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// COMPUTE kineticenergy without shared memory
////////////////////////////////////////////////////////////////////////////////
// Reduction kernel for kinetic energy without shared memory usage
template <class T>
__global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
cuda_compute_kineticenergy_withoutsharedmemory_KERNEL(T *g_idata, T *g_odata, unsigned int numberOfThreads)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    double store1, store2;
    if (idx >= numberOfThreads) return ;
    store1 = g_idata[idx];
    store2 = g_idata[idx+numberOfThreads];
    store1 *= store1;
    store2 *= store2;
    g_odata[idx] = store1 + store2;
    if(idx==0)g_odata[numberOfThreads] = 0.0;
}
// code to call above kernel
double cuda_compute_kineticenergy_withoutsharedmemory(int nd, int np, double mass){
    if( np%2 != 0)
    {
        printf("\n\n>>>>> ERROR: Numpber of molecules (np) is %d. Please use even number of molecules.\n\n", np);
        exit(-1);
    };
    int numberOfThreads = (nd*np)/2;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
    dim3 dimGrid;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    dimGrid = dim3(numBlocks, 1, 1);
    cuda_compute_kineticenergy_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(vel_gpu, parforce_gpu, numberOfThreads);
    CUT_CHECK_ERROR("cuda_compute_kineticenergy_withoutsharedmemory_KERNEL failed while profiling"); 
    numberOfThreads = numberOfThreads/2 + (numberOfThreads % 2 == 0 ? 0 : 1) ;
    numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    dimGrid = dim3(numBlocks, 1, 1);
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu, parforce_gpu, numberOfThreads);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_compute_kineticenergy_withoutsharedmemory reduce_withoutsharedmemory_KERNEL failed while profiling"); 
    return 0.0;
    #endif
    
    dimGrid = dim3(numBlocks, 1, 1);
    cuda_compute_kineticenergy_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(vel_gpu, parforce_gpu, numberOfThreads);
    CUT_CHECK_ERROR("cuda_compute_kineticenergy_withoutsharedmemory_KERNEL failed"); 
    numberOfThreads = numberOfThreads/2 + (numberOfThreads % 2 == 0 ? 0 : 1) ;
    numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    while(numberOfThreads>256){  
        dimGrid = dim3(numBlocks, 1, 1);
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock >>>(parforce_gpu, parforce_gpu, numberOfThreads);
        numberOfThreads = numberOfThreads/2 + (numberOfThreads % 2 == 0 ? 0 : 1) ;
        numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    }
    
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_compute_kineticenergy_withoutsharedmemory reduce_withoutsharedmemory_KERNEL failed"); 
    
    numberOfThreads *= 2;
    double* parforce = (double*) malloc(numberOfThreads * sizeof (double));    
    readArrayFromGPU_parforce(parforce,numberOfThreads );
    double sum = 0.0;
    for(int i=0; i<numberOfThreads; i++){
    	  sum += parforce[i];
    }
    
    return 0.5 * mass * sum;
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

///////////////////////////////////////////////////////////////////////// [K4sh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// COMPUTE kineticenergy with shared memory
////////////////////////////////////////////////////////////////////////////////
// Reduction kernel for kinetic energy with shared memory usage
template <class T>
__global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
cuda_compute_kineticenergy_withsharedmemory_KERNEL(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    
    __syncthreads();
    
    sdata[tid] *= sdata[tid];

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
// code to call above kernel
double cuda_compute_kineticenergy_withsharedmemory(int nd, int np, double mass){
    int numberOfThreads = nd*np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
    dim3 dimGrid;
    int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(double) : BLOCK_SIZE * sizeof(double);
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    dimGrid = dim3(numBlocks, 1, 1);
    cuda_compute_kineticenergy_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(vel_gpu, parforce_gpu, numberOfThreads);
    CUT_CHECK_ERROR("cuda_compute_kineticenergy_withsharedmemory_KERNEL failed"); 
    numberOfThreads = numBlocks;
    numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    dimGrid = dim3(numBlocks, 1, 1);
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu, parforce_gpu, numberOfThreads);
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_compute_kineticenergy_withsharedmemory reduce_withsharedmemory_KERNEL failed while profiling"); 
    return 0.0;
    #endif
    
    dimGrid = dim3(numBlocks, 1, 1);
    cuda_compute_kineticenergy_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(vel_gpu, parforce_gpu, numberOfThreads);
    CUT_CHECK_ERROR("cuda_compute_kineticenergy_withsharedmemory_KERNEL failed"); 
    numberOfThreads = numBlocks;
    numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
       
    while(numBlocks>1){  
        dimGrid = dim3(numBlocks, 1, 1);
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize >>>(parforce_gpu, parforce_gpu, numberOfThreads);
        numberOfThreads = numBlocks;
        numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    }
    
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_compute_kineticenergy_withsharedmemory reduce_withsharedmemory_KERNEL failed"); 
    
    double* parforce = (double*) malloc(numberOfThreads * sizeof (double));    
    readArrayFromGPU_parforce(parforce,numberOfThreads );
    double sum = 0.0;
    for(int i=0; i<numberOfThreads; i++){
    	  sum += parforce[i];
    }
    
    return 0.5 * mass * sum;
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

/////////////////////////////////////////////////////////////////////////// [K5]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// UPDATE pos
////////////////////////////////////////////////////////////////////////////////
// kernel code
static __global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
cuda_update_pos_KERNEL(int numberOfThreads, double dt, double * pos_gpu, const double * vel_gpu, const double * acc_gpu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if (idx >= numberOfThreads) return ;
    pos_gpu[idx] += vel_gpu[idx] * dt + 0.5 * acc_gpu[idx] * dt * dt;
} ;
// call to kernel code
void cuda_update_pos(int nd, int np, double dt)
{
    int numberOfThreads = nd*np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
        
    dim3 dimGrid(numBlocks) ;
    dim3 dimBlock(BLOCK_SIZE) ;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    cuda_update_pos_KERNEL<<<dimGrid,dimBlock>>>(numberOfThreads, dt, pos_gpu, vel_gpu, acc_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_update_pos_KERNEL failed while profiling"); 
    return;
    #endif

    cuda_update_pos_KERNEL<<<dimGrid,dimBlock>>>(numberOfThreads, dt, pos_gpu, vel_gpu, acc_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_update_pos_KERNEL failed");
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

/////////////////////////////////////////////////////////////////////////// [K6]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// UPDATE vel
////////////////////////////////////////////////////////////////////////////////
// kernel code
static __global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
cuda_update_vel_KERNEL(int numberOfThreads, double dt, double rmass, double * vel_gpu, const double * force_gpu, const double * acc_gpu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if (idx >= numberOfThreads) return ;
    vel_gpu[idx] += 0.5 * dt * (force_gpu[idx] * rmass + acc_gpu[idx]);
} ;
// call to kernel code
void cuda_update_vel(int nd, int np, double dt, double rmass)
{
    int numberOfThreads = nd*np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
        
    dim3 dimGrid(numBlocks) ;
    dim3 dimBlock(BLOCK_SIZE) ;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    cuda_update_vel_KERNEL<<<dimGrid,dimBlock>>>(numberOfThreads, dt, rmass, vel_gpu, force_gpu, acc_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_update_vel_KERNEL failed while profiling"); 
    return;
    #endif

    cuda_update_vel_KERNEL<<<dimGrid,dimBlock>>>(numberOfThreads, dt, rmass, vel_gpu, force_gpu, acc_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_update_vel_KERNEL failed");
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

/////////////////////////////////////////////////////////////////////////// [K7]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////// UPDATE acc
////////////////////////////////////////////////////////////////////////////////
// kernel code
static __global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
cuda_update_acc_KERNEL(int numberOfThreads, double rmass, double * acc_gpu, const double * force_gpu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if (idx >= numberOfThreads) return ;
    acc_gpu[idx] = force_gpu[idx] * rmass;
} ;
// call to kernel code
void cuda_update_acc(int nd, int np, double rmass)
{
    int numberOfThreads = nd*np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
        
    dim3 dimGrid(numBlocks) ;
    dim3 dimBlock(BLOCK_SIZE) ;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    cuda_update_acc_KERNEL<<<dimGrid,dimBlock>>>(numberOfThreads, rmass, acc_gpu, force_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_update_acc_KERNEL failed while profiling"); 
    return;
    #endif

    cuda_update_acc_KERNEL<<<dimGrid,dimBlock>>>(numberOfThreads, rmass, acc_gpu, force_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_update_acc_KERNEL failed");
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

//////////////////////////////////////////////////////////////////////// [K567M]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// UPDATE pos_vel_acc (this code removes artificial dependency and merges K5, K6 and K7)
////////////////////////////////////////////////////////////////////////////////
// kernel code
static __global__ void
#if TESTING_BLOCKPERSMUSAGE
__launch_bounds__(LAUNCHBOUND_MAXTHREADS,LAUNCHBOUND_MINBLOCKS)
#endif
cuda_mergedupdate_pos_vel_acc_KERNEL(int numberOfThreads, double dt, double rmass, double * pos_gpu, double * velnew_gpu, double * accnew_gpu, const double * vel_gpu, const double * acc_gpu, const double * force_gpu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if (idx >= numberOfThreads) return ;
    pos_gpu[idx] += vel_gpu[idx] * dt + 0.5 * acc_gpu[idx] * dt * dt;
    velnew_gpu[idx] += 0.5 * dt * (force_gpu[idx] * rmass + acc_gpu[idx]);
    accnew_gpu[idx] = force_gpu[idx] * rmass;
} ;
// call to kernel code
void cuda_mergedupdate_pos_vel_acc(int nd, int np, double dt, double rmass)
{
    int numberOfThreads = nd*np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
        
    dim3 dimGrid(numBlocks) ;
    dim3 dimBlock(BLOCK_SIZE) ;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    cuda_mergedupdate_pos_vel_acc_KERNEL<<<dimGrid,dimBlock>>>(numberOfThreads, dt, rmass, pos_gpu, velnew_gpu, accnew_gpu, vel_gpu, acc_gpu, force_gpu) ;
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_mergedupdate_pos_vel_acc_KERNEL failed while profiling"); 
    return;
    #endif

    cuda_mergedupdate_pos_vel_acc_KERNEL<<<dimGrid,dimBlock>>>(numberOfThreads, dt, rmass, pos_gpu, velnew_gpu, accnew_gpu, vel_gpu, acc_gpu, force_gpu) ;
    cudaThreadSynchronize();
    cudaError_t err = cudaMemcpy(vel_gpu, velnew_gpu, numberOfThreads*sizeof(double), cudaMemcpyDeviceToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: cuda_mergedupdate_pos_vel_acc_KERNEL copyDeviceToDevice_vel FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
    err = cudaMemcpy(acc_gpu, accnew_gpu, numberOfThreads*sizeof(double), cudaMemcpyDeviceToDevice);
    if( cudaSuccess != err)
    {
        printf("\n\nERROR: cuda_mergedupdate_pos_vel_acc_KERNEL copyDeviceToDevice_acc FAILED ||| %s\n\n", cudaGetErrorString(err));
        exit(-1);
    };
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_mergedupdate_pos_vel_acc_KERNEL failed");
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

//////////////////////////////////////////////////////////////////////// [K567S]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// UPDATE pos_vel_acc sequentially
////////////////////////////////////////////////////////////////////////////////
// call sequentially to update pos, vel and acc
void cuda_sequentialupdate_pos_vel_acc(int nd, int np, double dt, double rmass)
{
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_sequentialupdate_pos_vel_acc failed while profiling"); 
    return;
    #endif
    cuda_update_pos(nd, np, dt);
	cuda_update_vel(nd, np, dt, rmass);
	cuda_update_acc(nd, np, rmass);
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

/////////////////////////////////////////////////////////////////////// [K23Ssh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////// CUMULATE parpot and parforce (using sequential kernel launch for FERMI
/////////////////////////////////////////////////// hardware with shared memory)
double cuda_sequentiallaunch_withsharedmemory_cumulate_parpot_and_parforce(int nd, int np, int currentMoleculeIndex)
{
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_sequentiallaunch_withsharedmemory_cumulate_parpot_and_parforce failed while profiling"); 
    return 0.0;
    #endif
    cuda_cumulate_parforce_withsharedmemory(nd,np,currentMoleculeIndex); 
    return cuda_cumulate_parpot_withsharedmemory(nd, np);
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

////////////////////////////////////////////////////////////////////// [K23Snsh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////// CUMULATE parpot and parforce (using sequential kernel launch for FERMI
//////////////////////////////////////////////// hardware without shared memory)
double cuda_sequentiallaunch_withoutsharedmemory_cumulate_parpot_and_parforce(int nd, int np, int currentMoleculeIndex)
{
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_sequentiallaunch_withoutsharedmemory_cumulate_parpot_and_parforce failed while profiling"); 
    return 0.0;
    #endif
    cuda_cumulate_parforce_withoutsharedmemory(nd,np,currentMoleculeIndex); 
    return cuda_cumulate_parpot_withoutsharedmemory(nd, np);
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////// Create and destroy streams and events for parallel kernel launch and
//////////////////////////////////////////////////////////// check fermi support
void checkFermiSupport(){
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cutilSafeCall( cudaGetDevice(&cuda_device));	
    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, cuda_device) );
    if( (deviceProp.concurrentKernels == 0 )){
        printf("\n>>> WARNING: GPU does not support concurrent kernel execution\n    any parallel kernel runs will be serialized.\n");
    }else{
        printf("\n>>> Parallel Kernel launch supported by GPU.\n");
    }
    printf(">>> Detected Compute SM %d.%d hardware with %d multi-processors.\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount); 
}
cudaStream_t *streams;
void createStreamsForParallelKernelLaunch(int nstreams){
    streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    for(int i = 0; i < nstreams; i++)
        cutilSafeCall( cudaStreamCreate(&(streams[i])) );
}
void destroyStreamsForParallelKernelLaunch(int nstreams){
    for(int i = 0; i < nstreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/


/////////////////////////////////////////////////////////////////////// [K23Psh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////// CUMULATE parpot and parforce (using parallel kernel launch for FERMI
/////////////////////////////////////////////////// hardware with shared memory)
double cuda_parallellaunch_withsharedmemory_cumulate_parpot_and_parforce(int nd, int np, int currentMoleculeIndex)
{
    int numberOfThreads = np;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
    dim3 dimGrid;
    int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(double) : BLOCK_SIZE * sizeof(double);
    
    //int nkernels, nstreams = 0;  

    // allocate and initialize an array of stream handles
    //nkernels = 4;
    //nstreams = 4;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_parallellaunch_withsharedmemory_cumulate_parpot_and_parforce failed while profiling"); 
    return 0.0;
    #endif
    
    while(numBlocks>1){  
        //cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
        //for(int i = 0; i < nstreams; i++)
        //    cutilSafeCall( cudaStreamCreate(&(streams[i])) );

        // create CUDA event handles
        //cudaEvent_t start_event, stop_event;
        //cutilSafeCall( cudaEventCreate(&start_event) );
        //cutilSafeCall( cudaEventCreate(&stop_event) );
        
        // the events are used for synchronization only and hence do not need to record timings
        // this also makes events not introduce global sync points when recorded which is critical to get overlap 
        //cudaEvent_t *kernelEvent;
        //kernelEvent = (cudaEvent_t*) malloc(nkernels * sizeof(cudaEvent_t));
        //for(int i = 0; i < nkernels; i++)
        //    cutilSafeCall( cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming) );
        
        //cutilSafeCall( cudaEventRecord(start_event, 0) );
        
        dimGrid = dim3(numBlocks, 1, 1);
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize, streams[0] >>>(parpot_gpu, parpot_gpu, numberOfThreads);
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize, streams[1] >>>(parforce_gpu, parforce_gpu, numberOfThreads);
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize, streams[2] >>>(parforce_gpu+(np), parforce_gpu+(np), numberOfThreads);
        reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize, streams[3] >>>(parforce_gpu+(np*2), parforce_gpu+(np*2), numberOfThreads);  
        numberOfThreads = numBlocks;
        numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
        
        // in this sample we just wait until the GPU is done
        //cutilSafeCall( cudaEventRecord(stop_event, 0) );
        //cutilSafeCall( cudaEventSynchronize(stop_event) ); 
            
        // release resources
        //for(int i = 0; i < nkernels; i++) {
    	//	cudaStreamDestroy(streams[i]);
    	//	cudaEventDestroy(kernelEvent[i]);
        //}
        //free(streams);
        //free(kernelEvent);
        //cudaEventDestroy(start_event);
        //cudaEventDestroy(stop_event);
        //cudaThreadExit();
    }
        
    
    //nkernels = 3;
    //nstreams = 3;
        
    //cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    //for(int i = 0; i < nstreams; i++)
    //    cutilSafeCall( cudaStreamCreate(&(streams[i])) );

    // create CUDA event handles
    //cudaEvent_t start_event, stop_event;
    //cutilSafeCall( cudaEventCreate(&start_event) );
    //cutilSafeCall( cudaEventCreate(&stop_event) );
        
    // the events are used for synchronization only and hence do not need to record timings
    // this also makes events not introduce global sync points when recorded which is critical to get overlap 
    //cudaEvent_t *kernelEvent;
    //kernelEvent = (cudaEvent_t*) malloc(nkernels * sizeof(cudaEvent_t));
    //for(int i = 0; i < nkernels; i++)
    //    cutilSafeCall( cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming) );
        
    //cutilSafeCall( cudaEventRecord(start_event, 0) );
     
    dimGrid = dim3(numBlocks, 1, 1); 
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize, streams[0]  >>>(parforce_gpu, force_gpu+currentMoleculeIndex, numberOfThreads);
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize, streams[1]  >>>(parforce_gpu+(np), force_gpu+(np+currentMoleculeIndex), numberOfThreads);
    reduce_withsharedmemory_KERNEL<double><<< dimGrid, dimBlock, smemSize, streams[2]  >>>(parforce_gpu+(np*2), force_gpu+(np+np+currentMoleculeIndex), numberOfThreads); 
    
    // in this sample we just wait until the GPU is done
    //cutilSafeCall( cudaEventRecord(stop_event, 0) );
    //cutilSafeCall( cudaEventSynchronize(stop_event) ); 
            
    // release resources
    //for(int i = 0; i < nkernels; i++) {
    //	cudaStreamDestroy(streams[i]);
    //	cudaEventDestroy(kernelEvent[i]);
    //}
    //free(streams);
    //free(kernelEvent);
    //cudaEventDestroy(start_event);
    //cudaEventDestroy(stop_event);
    //cudaThreadExit(); 
    
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_parallellaunch_cumulate_parpot_and_parforce reduce_withsharedmemory_KERNEL failed"); 
    
    double* parpot = (double*) malloc(numberOfThreads * sizeof (double));    
    readArrayFromGPU_parpot(parpot,numberOfThreads );
    double sum = 0.0;
    for(int i=0; i<numberOfThreads; i++){
    	  sum += parpot[i];
    }
    
    return sum;
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/

/////////////////////////////////////////////////////////////////////// [K23Pnsh]
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////// CUMULATE parpot and parforce (using parallel kernel launch for FERMI
//////////////////////////////////////////////// hardware without shared memory)
double cuda_parallellaunch_withoutsharedmemory_cumulate_parpot_and_parforce(int nd, int np, int currentMoleculeIndex)
{
    if( np%2 != 0)
    {
        printf("\n\n>>>>> ERROR: Numpber of molecules (np) is %d. Please use even number of molecules.\n\n", np);
        exit(-1);
    };
    int numberOfThreads = np/2;
    int numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;
    
    dim3 dimBlock = dim3(BLOCK_SIZE, 1, 1);
    dim3 dimGrid;
    
    #if PROFILING
    cuda_dummy_KERNEL<<<1,1,1>>>();
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_parallellaunch_withoutsharedmemory_cumulate_parpot_and_parforce failed while profiling"); 
    return 0.0;
    #endif
    
    while(numberOfThreads>1){    
        dimGrid = dim3(numBlocks, 1, 1); 
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock, 0, streams[0]  >>>(parpot_gpu, parpot_gpu, numberOfThreads);
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock, 0, streams[1]  >>>(parforce_gpu, parforce_gpu, numberOfThreads);
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock, 0, streams[2]  >>>(parforce_gpu+(np), parforce_gpu+(np), numberOfThreads);
        reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock, 0, streams[3]  >>>(parforce_gpu+(np*2), parforce_gpu+(np*2), numberOfThreads);  
        numberOfThreads = numberOfThreads/2 + (numberOfThreads % 2 == 0 ? 0 : 1) ;
        numBlocks = numberOfThreads / BLOCK_SIZE + (numberOfThreads % BLOCK_SIZE == 0 ? 0 : 1) ;  	
    }
    
    dimGrid = dim3(numBlocks, 1, 1); 
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock, 0, streams[0]  >>>(parforce_gpu, force_gpu+currentMoleculeIndex, numberOfThreads);
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock, 0, streams[1]  >>>(parforce_gpu+(np), force_gpu+(np+currentMoleculeIndex), numberOfThreads);
    reduce_withoutsharedmemory_KERNEL<double><<< dimGrid, dimBlock, 0, streams[2]  >>>(parforce_gpu+(np*2), force_gpu+(np+np+currentMoleculeIndex), numberOfThreads);  
    
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_CHECK_ERROR("cuda_cumulate_parforce_withoutsharedmemory reduce_withoutsharedmemory_KERNEL failed"); 
    
    numberOfThreads *=2;
    double* parpot = (double*) malloc(numberOfThreads * sizeof (double));    
    readArrayFromGPU_parpot(parpot,numberOfThreads);
    double sum = 0.0;
    for(int i=0; i<numberOfThreads; i++){
    	  sum += parpot[i];
    }
    
    return sum;    
}
/******************************************************************************/
/****************************          END         ****************************/
/******************************************************************************/
