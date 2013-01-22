//################################################################################
//#
//# Author: Praveen Baburao Kulkarni praveen@spikingneurons.com
//#
//################################################################################
//#
//# 
//#
//################################################################################

/*
 * File:   main.c
 * Author: root
 *
 * Created on March 7, 2011, 1:17 PM
 */


#include "MD_define_v7_0.h"

#if MAINCODE 

double* acc;
double* box;
double* force;
double* pos;
double* vel;

void timestamp();
void initialize();
void compute();
double compute_distance(int moleculeindexi, int moleculeindexj, double* rij);
void compute_forceonparticle(int moleculeindexi, double distance, double truncated_distance, double* rij);
double compute_kineticenergy();
void update();

int main(int argc, char** argv) {
    np = ENTERED_NP;
    step_num = ENTERED_NUMOFSTEPS;
    acc = (double*) malloc(nd * np * sizeof (double));
    box = (double*) malloc(nd * sizeof (double));
    force = (double*) malloc(nd * np * sizeof (double));
    pos = (double*) malloc(nd * np * sizeof (double));
    vel = (double*) malloc(nd * np * sizeof (double));
    
    printf("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
    
    #if CUDA_CODE
    BLOCK_SIZE=ENTERED_BLOCKSIZE;
    // allocate memory on GPU
    allocateArrayOnGPU_acc(nd * np);
    allocateArrayOnGPU_force(nd * np);
    allocateArrayOnGPU_vel(nd * np);
    allocateArrayOnGPU_pos(nd * np);
    allocateArrayOnGPU_parpot(np);
    allocateArrayOnGPU_parforce(nd*np); 
    allocateArrayOnGPU_velnew(nd * np); 
    allocateArrayOnGPU_accnew(nd * np);
    #if FERMI_CODE
    printf("\n[ ______________ USING GPGPU WITH FERMI SUPPORT (Block size: %d) ______________ ]\n",BLOCK_SIZE);
    #else
    printf("\n[ ______________ USING GPGPU WITHOUT FERMI SUPPORT (Block size: %d) ______________ ]\n",BLOCK_SIZE);
    #endif
    #else
    printf("\n[ ______________ NOT USING GPGPU ______________ ]\n");
    #endif
    
    checkFermiSupport();

    timestamp();
    printf("\nA molecular dynamics program.");
    printf("\n    NP, the number of particles in the simulation is: %d", np);
    printf("\n    STEP_NUM, the number of time steps, is          : %d", step_num);
    printf("\n    DT, the size of each time step, is              : %E", dt);

    // Set the dimensions of the box
    int index = nd;
    while (index != 0) {
        box[--index] = 10.0;
    }

    // Set initial positions, velocities, and accelerations
    printf("\n\nInitializing positions, velocities, and accelerations.");
    initialize();


    // Compute the forces and energies.
    printf("\n\nComputing initial forces and energies.");
    compute();

    // Save the initial total energy for use in the accuracy check.
    e0 = potential + kinetic;

    printf("\n");
    printf("\n    At each step, we report the potential and kinetic energies.");
    printf("\n    The sum of these energies should be a constant.");
    printf("\n    As an accuracy check, we also print the relative error");
    printf("\n    in the total energy. (Total energy is: %E)", e0);
    printf("\n");
    printf("\n        Step      Potential       Kinetic        (P+K-E0)/E0");
    printf("\n                  Energy P        Energy K       Relative Energy Error");
    printf("\n");


    //  This is the main time stepping loop:
    //    Compute forces and energies,
    //    Update positions, velocities, accelerations.
    step_print = 0;
    step_print_index = 0;
    step_print_num = 10;

    step = 0;
    printf("\n        %d      %E       %E       %E", step, potential, kinetic, ((potential + kinetic - e0) / e0));
    step_print_index = step_print_index + 1;
    step_print = (step_print_index * step_num) / step_print_num;


    //
    gettimeofday(&ctime1,NULL);

    for (step = 1; step <= step_num; step++) {
        compute();
        // if ( step == step_print ){
        printf("\n        %d      %E       %E       %E", step, potential, kinetic, ((potential + kinetic - e0) / e0));
        //      step_print_index = step_print_index + 1;
        //      step_print = (step_print_index * step_num) / step_print_num;
        // }
        update();
    }

    //
    gettimeofday(&ctime2,NULL);
    
    long timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);

    printf("\n");
    printf("\n    Elapsed cpu time for main computation: %ld microsecond", timeTaken);
    printf("\n");
    printf("\nMD");
    printf("\n    Normal end of execution.");
    printf("\n");
    timestamp();

    #if CUDA_CODE
    // free memory from GPU and destroy streams
    freeArrayFromGPU_acc();
    freeArrayFromGPU_force();
    freeArrayFromGPU_vel();
    freeArrayFromGPU_pos();
    freeArrayFromGPU_parpot();
    freeArrayFromGPU_parforce();
    freeArrayFromGPU_velnew();
    freeArrayFromGPU_accnew();
    destroyStreamsForParallelKernelLaunch(4);
    #endif
    
    printf("\n\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n");

    return (EXIT_SUCCESS);
}

void timestamp() {
    time_t ltime;
    ltime = time(NULL);
    printf("\n%s", asctime(localtime(&ltime)));
}

//Parameters :
//Input, integer(kind = 4) NP, the number of particles.
//Input, integer(kind = 4) ND, the number of spatial dimensions.
//Input, real(kind = 8) BOX(ND), specifies the maximum position
//of particles in each dimension.
//
//Input / output, integer(kind = 4) SEED, a seed for the random
//number generator.
//Output, real(kind = 8) POS(ND, NP), the position of each particle.
//Output, real(kind = 8) VEL(ND, NP), the velocity of each particle.
//Output, real(kind = 8) ACC(ND, NP), the acceleration of each particle.

void initialize() {
    int i;
    int j;

    seed = 123456789;
    srand(seed);


    // Start by setting the positions to random numbers between 0 and 1

    //    while (index != 0) {
    //        pos[--index] = (double)rand()/((double)(RAND_MAX)+1.0);
    //    }

    double temp = 0.0;
    for (j = 0; j < np; j++) {
        pos[j] = temp;
        pos[j + np] = temp;
        pos[j + np + np] = temp;
        temp = temp + 0.00001;
    }

    //Use these random values as scale factors to pick random locations
    //inside the box.
    for (i = 0; i < nd; i++) {
        for (j = 0; j < np; j++) {
            int tempIndex = i * np + j;
            pos[tempIndex] = box[i] * pos[tempIndex];
        }
    }


    //Velocities and accelerations begin at 0
    for (i = 0; i < nd; i++) {
        for (j = 0; j < np; j++) {
            int tempIndex = i * np + j;
            vel[tempIndex] = 0.0;
            acc[tempIndex] = 0.0;
        }
    }
    
    #if CUDA_CODE
    // copy data to GPU
    copyArrayOnGPU_acc(acc, nd * np);
    copyArrayOnGPU_force(force, nd * np);
    copyArrayOnGPU_vel(vel, nd * np);
    copyArrayOnGPU_pos(pos, nd * np);
    createStreamsForParallelKernelLaunch(4);
    #endif 
}



//COMPUTE computes the forces and energies.
//
//  Discussion:
//    The computation of forces and energies is fully parallel.
//    The potential function V(X) is a harmonic well which smoothly
//    saturates to a maximum value at PI/2:
//      v(x) = ( sin ( min ( x, PI2 ) ) )**2
//    The derivative of the potential is:
//      dv(x) = 2.0D+00 * sin ( min ( x, PI2 ) ) * cos ( min ( x, PI2 ) )
//            = sin ( 2.0 * min ( x, PI2 ) )
//
//  Parameters:
//    Input, integer ( kind = 4 ) NP, the number of particles.
//    Input, integer ( kind = 4 ) ND, the number of spatial dimensions.
//    Input, real ( kind = 8 ) POS(ND,NP), the position of each particle.
//    Input, real ( kind = 8 ) VEL(ND,NP), the velocity of each particle.
//    Input, real ( kind = 8 ) MASS, the mass of each particle.
//    Output, real ( kind = 8 ) F(ND,NP), the forces.
//    Output, real ( kind = 8 ) POT, the total potential energy.
//    Output, real ( kind = 8 ) KIN, the total kinetic energy.

void compute() {
    potential = 0.0;
    int i;
    
    #if !CUDA_CODE
    double* rij;
    double d;
    double d2;
    int j, i1;
    rij = (double*) malloc(nd * sizeof (double));
    #endif
    
    for (i = 0; i < np; i++) {
    
    	  #if CUDA_CODE
    	      // call kernels
        	  cuda_compute_forceonparticle(nd, np, i, PI2);  
        	  #if FERMI_CODE
        	  potential += cuda_parallellaunch_withoutsharedmemory_cumulate_parpot_and_parforce(nd,np,i); 
        	  #else
        	  potential += cuda_sequentiallaunch_withoutsharedmemory_cumulate_parpot_and_parforce(nd,np,i);  
        	  #endif 
    	  #else
    	  
        // Compute the potential energy and forces
        for (i1 = 0; i1 < nd; i1++) {
            force[i1 * np + i] = 0.0;
        }

        for (j = 0; j < np; j++) {

            if (i != j) {

                // compute_distance
                d = compute_distance(i, j, rij);

                // Truncate the distance ...  min ( d, PI2 )
                d2 = (d < PI2) ? d : PI2;

                /** This is U shaped energy. Lennard-Jones Potential could be
                 *  another module
                 **/

                // Attribute half of the total potential energy to particle J.
                potential += 0.5 * sin(d2) * sin(d2); // Discuss on potential calculation            

                // Add particle J's contribution to the force on particle I.
                compute_forceonparticle(i, d, d2, rij);
            }
        }
        
        #endif
        
    }
    
    // Compute the total kinetic energy.
    #if CUDA_CODE    
    
    kinetic = cuda_compute_kineticenergy_withoutsharedmemory(nd, np, mass);
    
    #else
    
    kinetic = compute_kineticenergy();
    
    #endif
    
    
    
}


#if !CUDA_CODE

double compute_distance(int moleculeindexi, int moleculeindexj, double* rij) {
    double d = 0.0;
    int j1 = 0;
    for (j1 = 0; j1 < nd; j1++) {
        rij[j1] = pos[j1 * np + moleculeindexi] - pos[j1 * np + moleculeindexj];
        d = d + pow(rij[j1], 2.0);
    }
    return sqrt(d);
}

void compute_forceonparticle(int moleculeindexi, double distance, double truncated_distance, double* rij) {
    for (int j1 = 0; j1 < nd; j1++) {
        force[j1 * np + moleculeindexi] = force[j1 * np + moleculeindexi] - rij[j1] * sin(2.0 * truncated_distance) / distance;
    }
}

double compute_kineticenergy() {
    int j1 = 0;
    double temp = 0.0;
    for (j1 = 0; j1 < nd * np; j1++) {
        temp = temp + pow(vel[j1], 2.0);
    }
    return 0.5 * mass * temp;
}

#endif

// UPDATE updates positions, velocities and accelerations.
//
//  Discussion:
//    The time integration is fully parallel.
//    A velocity Verlet algorithm is used for the updating.
//    x(t+dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt * dt
//    v(t+dt) = v(t) + 0.5 * ( a(t) + a(t+dt) ) * dt
//    a(t+dt) = f(t) / m
//
//  Parameters:
//    Input, integer ( kind = 4 ) NP, the number of particles.
//    Input, integer ( kind = 4 ) ND, the number of spatial dimensions.
//    Input/output, real ( kind = 8 ) POS(ND,NP), the position of each particle.
//    Input/output, real ( kind = 8 ) VEL(ND,NP), the velocity of each particle.
//    Input, real ( kind = 8 ) F(ND,NP), the force on each particle.
//    Input/output, real ( kind = 8 ) ACC(ND,NP), the acceleration of each
//    Input, real ( kind = 8 ) MASS, the mass of each particle.
//    Input, real ( kind = 8 ) DT, the time step.

void update() {

    double rmass;
    rmass = 1.0 / mass;

    #if CUDA_CODE
    
    // call kernels
    cuda_update_pos(nd, np, dt);
    cuda_update_vel(nd, np, dt, rmass);
    cuda_update_acc(nd, np, rmass);
    
    //cuda_update_pos_vel_acc(nd, np, dt, rmass);
    
    #else
    
    int i, j;
    for (j = 0; j < np; j++) {
        for (i = 0; i < nd; i++) {
            pos[i * np + j] = pos[i * np + j] + vel[i * np + j] * dt + 0.5 * acc[i * np + j] * dt * dt; // Kernel 1
            vel[i * np + j] = vel[i * np + j] + 0.5 * dt * (force[i * np + j] * rmass + acc[i * np + j]); // Kernel 2
            acc[i * np + j] = force[i * np + j] * rmass; // Kernel 3
        }
    }
    
    #endif
}


#endif

