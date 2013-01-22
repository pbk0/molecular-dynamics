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

#if TESTING_KERNEL


double* acc;
double* box;
double* force;
double* pos;
double* vel;

void initialize();
void timestamp();
void testWithVariableBlockSizeAndMolecules(int numberOfIterations, int numberOfMolecules);

int main(int argc, char** argv) {

    checkFermiSupport();

    int numberOfIterations = ENTERED_ITERATIONS;
        
    int numberOfMolecules = ENTERED_NP;
    
    while(1){
        testWithVariableBlockSizeAndMolecules(numberOfIterations, numberOfMolecules);
        numberOfMolecules += 1500;
        if(numberOfMolecules>349520){
            numberOfMolecules = 349520;
            testWithVariableBlockSizeAndMolecules(numberOfIterations, numberOfMolecules);
            break;
        }
    }
}



void timestamp() {
    time_t ltime;
    ltime = time(NULL);
    printf("\n%s", asctime(localtime(&ltime)));
}


void initialize() {
    int i;
    int j;

    seed = 123456789;
    srand(seed);


    // Start by setting the positions to random numbers between 0 and 1

    //    while (index != 0) {
    //    pos[--index] = (double)rand()/((double)(RAND_MAX)+1.0);
    //    }
    
    // Set the dimensions of the box
    int index = nd;
    while (index != 0) {
        box[--index] = 10.0;
    }

    double temp = 0.0;
    for (j = 0; j < np; j++) {
    pos[j] = temp;
    pos[j + np] = temp;
    pos[j + np + np] = temp;
    temp = temp + 0.0000001;
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
}

void testWithVariableBlockSizeAndMolecules(int numberOfIterations, int numberOfMolecules){
    
    //np=349520;
    np = numberOfMolecules;
    
    int count;
	double rmass = 1.0/mass; 
	
	long timeTaken, mcotime1, mcotime2 =0;

    
    
    printf("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
    printf("\n|||||||||||||||||||||||||||||||\tREPORT\titerations:\t%d\tmolecules:\t%d",numberOfIterations , numberOfMolecules);
    printf("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
    timestamp();
    printf("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
    printf("\n");
    
    acc = (double*) malloc(nd * np * sizeof (double));
    box = (double*) malloc(nd * sizeof (double));
    force = (double*) malloc(nd * np * sizeof (double));
    pos = (double*) malloc(nd * np * sizeof (double));
	vel = (double*) malloc(nd * np * sizeof (double));
	double* parpot = (double*) malloc(np * sizeof (double));
    double* parforce = (double*) malloc(nd * np * sizeof (double));
    double* velnew = (double*) malloc(nd * np * sizeof (double));
	double* accnew = (double*) malloc(nd * np * sizeof (double));
  
    
	// initialize the arrays with some values
	initialize();
	allocateArrayOnGPU_acc(nd * np);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_vel(nd * np);
	allocateArrayOnGPU_pos(nd * np);
	allocateArrayOnGPU_parpot(np);
	allocateArrayOnGPU_parforce(nd * np);   
    allocateArrayOnGPU_velnew(nd * np); 
    allocateArrayOnGPU_accnew(nd * np);
    copyArrayOnGPU_acc(acc, nd * np);
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	copyArrayOnGPU_pos(pos, nd * np);
	copyArrayOnGPU_parpot(parpot, np);
	copyArrayOnGPU_vel(vel, nd * np);
	copyArrayOnGPU_velnew(velnew, nd * np);
	copyArrayOnGPU_accnew(accnew, nd * np);
	readArrayFromGPU_acc(acc, nd * np);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	readArrayFromGPU_pos(pos, nd * np);
	readArrayFromGPU_parpot(parpot, np);
	readArrayFromGPU_vel(vel, nd * np);
	readArrayFromGPU_velnew(velnew, nd * np);
	readArrayFromGPU_accnew(accnew, nd * np);
    freeArrayFromGPU_acc();
	freeArrayFromGPU_force();
	freeArrayFromGPU_vel();
	freeArrayFromGPU_pos();
	freeArrayFromGPU_parpot();
	freeArrayFromGPU_parforce();
    freeArrayFromGPU_velnew();
    freeArrayFromGPU_accnew();
	
	// title
	printf("\n[K#]\t");
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    printf("\t[B%d]",BLOCK_SIZE);
		BLOCK_SIZE+=32;
	}
	printf("\t[MCO]\t");
	
	
	
	// [K1] kernel 1 : cuda_compute_forceonparticle(nd, np, 1, PI2);
	printf("\n[K1]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_pos(nd * np);
	allocateArrayOnGPU_parpot(np);
	allocateArrayOnGPU_parforce(nd * np);   
	copyArrayOnGPU_parforce(parforce, nd * np);
	copyArrayOnGPU_pos(pos, nd * np);
	copyArrayOnGPU_parpot(parpot, np);    
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    if(BLOCK_SIZE>384){
	        printf("\t---");
	    }else{
    	    gettimeofday(&ctime1,NULL);
    	    count = numberOfIterations;
    	    while(count--){
    	    cuda_compute_forceonparticle(nd, np, 1, PI2);
    	    }
    	    gettimeofday(&ctime2,NULL);
    	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
    	    printf("\t%ld",timeTaken);
	    }
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_parforce(parforce, nd * np);
	readArrayFromGPU_pos(pos, nd * np);
	readArrayFromGPU_parpot(parpot, np);
	freeArrayFromGPU_pos();
	freeArrayFromGPU_parpot();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K2sh] kernel 2 with shared memory : cuda_cumulate_parpot_withsharedmemory(nd,np);
	printf("\n[K2sh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_parpot(np);
	copyArrayOnGPU_parpot(parpot, np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_cumulate_parpot_withsharedmemory(nd,np);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_parpot(parpot, np);
	freeArrayFromGPU_parpot();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K2nsh] kernel 2 without shared memory : cuda_cumulate_parpot_withoutsharedmemory(nd,np);
	printf("\n[K2nsh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_parpot(np);
	copyArrayOnGPU_parpot(parpot, np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_cumulate_parpot_withoutsharedmemory(nd,np);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_parpot(parpot, np);
	freeArrayFromGPU_parpot();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K3sh] kernel 3  with shared memory : cuda_cumulate_parforce_withsharedmemory(nd,np,1);
	printf("\n[K3sh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_parforce(nd * np);   
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_cumulate_parforce_withsharedmemory(nd,np,1);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	freeArrayFromGPU_force();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K3nsh] kernel 3  without shared memory : cuda_cumulate_parforce_withoutsharedmemory(nd,np,1);
	printf("\n[K3nsh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_parforce(nd * np);   
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_cumulate_parforce_withoutsharedmemory(nd,np,1);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	freeArrayFromGPU_force();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K4sh] kernel 4 with shared memory : cuda_compute_kineticenergy_withsharedmemory(nd, np, mass);
	printf("\n[K4sh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_vel(nd * np);
	allocateArrayOnGPU_parforce(nd * np);   
    copyArrayOnGPU_vel(vel, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_compute_kineticenergy_withsharedmemory(nd, np, mass);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
    readArrayFromGPU_vel(vel, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	freeArrayFromGPU_vel();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K4nsh] kernel 4 without shared memory : cuda_compute_kineticenergy_withoutsharedmemory(nd, np, mass);
	printf("\n[K4nsh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_vel(nd * np);
	allocateArrayOnGPU_parforce(nd * np);   
    copyArrayOnGPU_vel(vel, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_compute_kineticenergy_withoutsharedmemory(nd, np, mass);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
    readArrayFromGPU_vel(vel, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	freeArrayFromGPU_vel();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K5] kernel 5 : cuda_update_pos(nd, np, dt);
	printf("\n[K5]\t");
	gettimeofday(&ctime1,NULL);
    allocateArrayOnGPU_acc(nd * np);
	allocateArrayOnGPU_vel(nd * np);
	allocateArrayOnGPU_pos(nd * np);
    copyArrayOnGPU_acc(acc, nd * np);
	copyArrayOnGPU_pos(pos, nd * np);
	copyArrayOnGPU_vel(vel, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_update_pos(nd, np, dt);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
    readArrayFromGPU_acc(acc, nd * np);
	readArrayFromGPU_pos(pos, nd * np);
	readArrayFromGPU_vel(vel, nd * np);
    freeArrayFromGPU_acc();
	freeArrayFromGPU_vel();
	freeArrayFromGPU_pos();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K6] kernel 6 : cuda_update_vel(nd, np, dt, rmass);
	printf("\n[K6]\t");
	gettimeofday(&ctime1,NULL);
	
    allocateArrayOnGPU_acc(nd * np);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_vel(nd * np);
    copyArrayOnGPU_acc(acc, nd * np);
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_vel(vel, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;       
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_update_vel(nd, np, dt, rmass);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
    readArrayFromGPU_acc(acc, nd * np);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_vel(vel, nd * np);
    freeArrayFromGPU_acc();
	freeArrayFromGPU_force();
	freeArrayFromGPU_vel();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K7] kernel 7 : cuda_update_acc(nd, np, rmass);
	printf("\n[K7]\t");
	gettimeofday(&ctime1,NULL);
    allocateArrayOnGPU_acc(nd * np);
	allocateArrayOnGPU_force(nd * np);
    copyArrayOnGPU_acc(acc, nd * np);
	copyArrayOnGPU_force(force, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_update_acc(nd, np, rmass);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
    readArrayFromGPU_acc(acc, nd * np);
	readArrayFromGPU_force(force, nd * np);
    freeArrayFromGPU_acc();
	freeArrayFromGPU_force();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	
	
	// [K567M] merged launch of kernals 5, 6 and 7 : cuda_mergedupdate_pos_vel_acc(nd, np, dt, rmass);
	printf("\n[K567M]\t");
	gettimeofday(&ctime1,NULL);
    allocateArrayOnGPU_acc(nd * np);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_vel(nd * np);
	allocateArrayOnGPU_pos(nd * np);  
    allocateArrayOnGPU_velnew(nd * np); 
    allocateArrayOnGPU_accnew(nd * np);
    copyArrayOnGPU_acc(acc, nd * np);
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_pos(pos, nd * np);
	copyArrayOnGPU_vel(vel, nd * np);
	copyArrayOnGPU_velnew(velnew, nd * np);
	copyArrayOnGPU_accnew(accnew, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_mergedupdate_pos_vel_acc(nd, np, dt, rmass);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
    readArrayFromGPU_acc(acc, nd * np);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_pos(pos, nd * np);
	readArrayFromGPU_vel(vel, nd * np);
	readArrayFromGPU_velnew(velnew, nd * np);
	readArrayFromGPU_accnew(accnew, nd * np);
    freeArrayFromGPU_acc();
	freeArrayFromGPU_force();
	freeArrayFromGPU_vel();
	freeArrayFromGPU_pos();
    freeArrayFromGPU_velnew();
    freeArrayFromGPU_accnew();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	
	// [K567S] sequential launch of three kernels 5,6 and 7 : cuda_sequentialupdate_pos_vel_acc(nd, np, dt, rmass);
	printf("\n[K567S]\t");
	gettimeofday(&ctime1,NULL);
    allocateArrayOnGPU_acc(nd * np);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_vel(nd * np);
	allocateArrayOnGPU_pos(nd * np);
    copyArrayOnGPU_acc(acc, nd * np);
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_pos(pos, nd * np);
	copyArrayOnGPU_vel(vel, nd * np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_sequentialupdate_pos_vel_acc(nd, np, dt, rmass);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
    readArrayFromGPU_acc(acc, nd * np);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_pos(pos, nd * np);
	readArrayFromGPU_vel(vel, nd * np);
    freeArrayFromGPU_acc();
	freeArrayFromGPU_force();
	freeArrayFromGPU_vel();
	freeArrayFromGPU_pos();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K23Ssh] sequential launch of two kernels 2 and 3 with shared memory : cuda_sequentiallaunch_withsharedmemory_cumulate_parpot_and_parforce(nd, np, 1);
	printf("\n[K23Ssh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_parpot(np);
	allocateArrayOnGPU_parforce(nd * np);
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	copyArrayOnGPU_parpot(parpot, np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_sequentiallaunch_withsharedmemory_cumulate_parpot_and_parforce(nd, np, 1);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	readArrayFromGPU_parpot(parpot, np);
	freeArrayFromGPU_force();
	freeArrayFromGPU_parpot();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K23Snsh] sequential launch of two kernels 2 and 3 without shared memory : cuda_sequentiallaunch_withoutsharedmemory_cumulate_parpot_and_parforce(nd, np, 1);
	printf("\n[K23Snsh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_parpot(np);
	allocateArrayOnGPU_parforce(nd * np);
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	copyArrayOnGPU_parpot(parpot, np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_sequentiallaunch_withoutsharedmemory_cumulate_parpot_and_parforce(nd, np, 1);
	    }
	    gettimeofday(&ctime2,NULL);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	readArrayFromGPU_parpot(parpot, np);
	freeArrayFromGPU_force();
	freeArrayFromGPU_parpot();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// [K23Psh] parallel launch of two kernels 2 and 3 with shared memory : cuda_parallellaunch_withsharedmemory_cumulate_parpot_and_parforce(nd, np, 1);
	printf("\n[K23Psh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_parpot(np);
	allocateArrayOnGPU_parforce(nd * np);
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	copyArrayOnGPU_parpot(parpot, np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    createStreamsForParallelKernelLaunch(4);
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_parallellaunch_withsharedmemory_cumulate_parpot_and_parforce(nd, np, 1);
	    }
	    gettimeofday(&ctime2,NULL);
	    destroyStreamsForParallelKernelLaunch(4);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	readArrayFromGPU_parpot(parpot, np);
	freeArrayFromGPU_force();
	freeArrayFromGPU_parpot();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);

	// [K23Pnsh] parallel launch of two kernels 2 and 3 with shared memory : cuda_parallellaunch_withoutsharedmemory_cumulate_parpot_and_parforce(nd, np, 1);
	printf("\n[K23Pnsh]\t");
	gettimeofday(&ctime1,NULL);
	allocateArrayOnGPU_force(nd * np);
	allocateArrayOnGPU_parpot(np);
	allocateArrayOnGPU_parforce(nd * np);
	copyArrayOnGPU_force(force, nd * np);
	copyArrayOnGPU_parforce(parforce, nd * np);
	copyArrayOnGPU_parpot(parpot, np);
	gettimeofday(&ctime2,NULL);
	mcotime1 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	BLOCK_SIZE = 32;    
	while(BLOCK_SIZE<=512){
	    createStreamsForParallelKernelLaunch(4);
	    gettimeofday(&ctime1,NULL);
	    count = numberOfIterations;
	    while(count--){
	    cuda_parallellaunch_withoutsharedmemory_cumulate_parpot_and_parforce(nd, np, 1);
	    }
	    gettimeofday(&ctime2,NULL);
	    destroyStreamsForParallelKernelLaunch(4);
	    timeTaken = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	    printf("\t%ld",timeTaken);
		BLOCK_SIZE+=32;
	}
	gettimeofday(&ctime1,NULL);
	readArrayFromGPU_force(force, nd * np);
	readArrayFromGPU_parforce(parforce, nd * np);
	readArrayFromGPU_parpot(parpot, np);
	freeArrayFromGPU_force();
	freeArrayFromGPU_parpot();
	freeArrayFromGPU_parforce();
	gettimeofday(&ctime2,NULL);
	mcotime2 = (ctime2.tv_sec*1000000+(long)ctime2.tv_usec)-(ctime1.tv_sec*1000000+(long)ctime1.tv_usec);
	printf("\t%ld",mcotime1 + mcotime2);
	
	// free host resourses
	free(acc);
	free(box);
	free(force);
	free(pos);
	free(vel);
	free(parpot);
	free(parforce);
	free(velnew);
	free(accnew);
		
	printf("\n\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
	timestamp();
	printf("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
	printf("\n************************************* END oF REPORT *********************************************\n\n");
}






#endif








