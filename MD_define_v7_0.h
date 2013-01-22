//################################################################################
//#
//# Author: Praveen Baburao Kulkarni praveen@spikingneurons.com
//#
//################################################################################
//#
//# 
//#
//################################################################################


#define MAINCODE 0
#if MAINCODE 
#define CUDA_CODE 
#define FERMI_CODE 
#define ENTERED_NUMOFSTEPS 
#define ENTERED_BLOCKSIZE 
#define ENTERED_NP   
#endif

#define TESTING_KERNEL 0
#if TESTING_KERNEL
#define ENTERED_ITERATIONS  
#define ENTERED_NP  
#endif

#define TESTING_BLOCKPERSMUSAGE 0
#if TESTING_BLOCKPERSMUSAGE
#define LAUNCHBOUND_MAXTHREADS    
#define LAUNCHBOUND_MINBLOCKS   
#define ENTERED_ITERATIONS1   
#define ENTERED_NP1     
#endif

#define PROFILING 0
#if PROFILING
#define ENTERED_BLOCKSIZE
#define ENTERED_NP 
#endif


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#include <cuda.h>
#include <cublas.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <shrUtils.h>
#include "stdio.h"
#include "MD_cuda_v7_0.cu"


const double PI2 = 3.141592653589793 / 2.0;
const int nd = 3;
int np;
timeval ctime1;
timeval ctime2;
double dt = 0.0001;
double e0;
double kinetic;
double mass = 1.0;
double potential;
int seed;
int step;
int step_num;
int step_print;
int step_print_index;
int step_print_num;





/*
 * Changes done:
 *
 * v1_1
 * 1] Removed random generator for molecule coordinates and replaced with static generator
 * 2] Bug in old version fixed which led to NaN
 *
 * v2_0
 * 1] Added these functions for modularity
 *      double compute_distance(int moleculeindexi, int moleculeindexj, double* rij);
 *      void compute_forceonparticle(int moleculeindexi, double distance, double truncated_distance, double* rij);
 *      double compute_kineticenergy();
 *
 * v3_0
 * 1] Implemented GPU code for memory allocation and error handling code
 * 2] CUDA implementation done for update() method
 * 
 * v3_1
 * 1] Modified the kernel to solve all dimensions simultaneously
 * 
 * v3_2
 * 1] CUDA implementation done for compute() method
 * 2] Finished implementation for CUDA
 * 3] Added 'MD_define.h' file to project to hold common variables and declarations  
 * 
 * v4_0
 * 1] Added 'MD_testkernel_v4_0.cu' to project
 * 2] Impleted test code for taking readings at different block sizes in 'MD_testkernel_v4_0.cu'
 * 
 * v4_1
 * 1] Impleted test code for taking readings for different number of molecules in 'MD_testkernel_v4_0.cu'
 *
 * v4_2
 * 1] testWithVariableBlockSizeAndMolecules() method implemented and replaced
 *
 * v7_0
 * 1] cuda_update_pos_vel_acc_KERNEL() Kernel added which works as a single update kernel with artificial dependency 
 *      removed but on cost of extra memory usage.
 *
 * v7_0
 * 1] cuda_parallel_launch_cumulate_parpot_and_parforce() Kernel added which will do work of cumulate parforce and 
 *      cumulate parpot in parallel by using parallel launch capability of fermi hardware
 * 2] Added Makefile to project to make use of project configuration setup used in NVIDIA_GPU_Computing_SDK tutorials
 * 3] Added CompileAndRun file to project for automated compile and launch in both normal execution mode and testing mode
 *
 * v7_0
 * 1] added MD_minblockpersmusage_v7_0.cu file to see perfromance with different blocks per SM
 */


