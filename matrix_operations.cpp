#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include "matrix_operations.h"

#define BLK_SIZE 128
#define min(a,b) (((a)<(b))?(a):(b))

//Function for initializing matrix with random values
void initialize_mat(double* A, int m, int n){
	int i, j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)	
			A[i*n+j] = rand() % 10;
}

// Naive Matrix multiplication method
double* naive_multiply(const double *A, const double *B, int m, int n, int p){
    int i, j, k;

	double *C;
	C = (double*)calloc(m * p, sizeof(double));

    for(i = 0; i < m; i++)
        for(j = 0; j < p; j++)
			for(k = 0; k < n; k++)
                C[p*i+j] += A[n*i+k] * B[p*k+j];	

	return C;		      
}

// Block Multiplication
double* multiply_block(const double *A, const double *B, int m, int n, int p){ 

	int i, j, k, ii, jj, kk, bs = BLK_SIZE;

	double *C;
	C = (double*)calloc(m * p, sizeof(double));

	for(ii = 0; ii < m; ii += bs)
		for(jj = 0; jj < p; jj += bs)
			for(kk = 0; kk < n; kk += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(j = jj; j < min(p, jj+bs); j++)
						for(k = kk; k < min(n, kk+bs); k++)
							C[p*i+j] += A[n*i+k] * B[p*k+j];
	return C;		      
											
}

// Multiplication with blocking and Loop reorder
double* multiply_block_reorder(const double *A, const double *B, int m, int n, int p){ 
	int i, j, k, ii, jj, kk, bs = BLK_SIZE;

	double *C;
	C = (double*)calloc(m * p, sizeof(double));

	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
						for(j = jj; j < min(p, jj+bs); j++)
							C[p*i+j] += A[n*i+k] * B[p*k+j];						
				
	return C;
}

// Multiplication with blocking, Loop reorder and reuse
 double* multiply_block_reorder_reuse(const double *A, const double *B, int m, int n, int p){ 
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;
	
	double *C;
	C = (double*)calloc(m * p, sizeof(double));

	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j++)
							C[p*i+j] += Aik * B[p*k+j];		
					}					
	return C;
}

// Multiplication with blocking, Loop reorder, reuse and unrolling
 double* multiply_block_reorder_reuse_unroll_2(const double *A, const double *B, int m, int n, int p){ 
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;

	double *C;
	C = (double*)calloc(m * p, sizeof(double));

	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j+=2)
						{
							C[p*i+j] += Aik * B[p*k+j];
							C[p*i+j+1] += Aik * B[p*k+j+1];		
						}
					}					
	return C;
}

// Multiplication with blocking, Loop reorder, reuse and unrolling
 double* multiply_block_reorder_reuse_unroll_4(const double *A, const double *B, int m, int n, int p){ 
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;

	double *C;
	C = (double*)calloc(m * p, sizeof(double));

	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j+=4)
						{
							C[p*i+j] += Aik * B[p*k+j];
							C[p*i+j+1] += Aik * B[p*k+j+1];		
							C[p*i+j+2] += Aik * B[p*k+j+2];	
							C[p*i+j+3] += Aik * B[p*k+j+3];		
						}
					}					
	return C;
}

// Multiplication with blocking, Loop reorder, reuse and unrolling
 double* multiply_block_reorder_reuse_unroll_8(  const double *A, const double *B, int m, int n, int p){ 
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;

	double *C;
	C = (double*)calloc(m * p, sizeof(double));


	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j+=8)
						{
							C[p*i+j] += Aik * B[p*k+j];
							C[p*i+j+1] += Aik * B[p*k+j+1];		
							C[p*i+j+2] += Aik * B[p*k+j+2];	
							C[p*i+j+3] += Aik * B[p*k+j+3];	
							C[p*i+j+4] += Aik * B[p*k+j+4];		
							C[p*i+j+5] += Aik * B[p*k+j+5];	
							C[p*i+j+6] += Aik * B[p*k+j+6];	
							C[p*i+j+7] += Aik * B[p*k+j+7];	
						}
					}					
	return C;
}

// Multiplication with blocking, Loop reorder, reuse and unrolling
 double* multiply_block_reorder_reuse_unroll_16(  const double *A, const double *B, int m, int n, int p){ 
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;

	double *C;
	C = (double*)calloc(m * p, sizeof(double));


	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j+=16)
						{
							C[p*i+j] += Aik * B[p*k+j];
							C[p*i+j+1] += Aik * B[p*k+j+1];		
							C[p*i+j+2] += Aik * B[p*k+j+2];	
							C[p*i+j+3] += Aik * B[p*k+j+3];	
							C[p*i+j+4] += Aik * B[p*k+j+4];		
							C[p*i+j+5] += Aik * B[p*k+j+5];	
							C[p*i+j+6] += Aik * B[p*k+j+6];	
							C[p*i+j+7] += Aik * B[p*k+j+7];
							C[p*i+j+8] += Aik * B[p*k+j+8];		
							C[p*i+j+9] += Aik * B[p*k+j+9];	
							C[p*i+j+10] += Aik * B[p*k+j+10];	
							C[p*i+j+11] += Aik * B[p*k+j+11];		
							C[p*i+j+12] += Aik * B[p*k+j+12];	
							C[p*i+j+13] += Aik * B[p*k+j+13];	
							C[p*i+j+14] += Aik * B[p*k+j+14];	
							C[p*i+j+15] += Aik * B[p*k+j+15];	
						}
					}					
	return C;
}

// Multiplciation helper
 double* multiply_helper(const double *A, const double *B, int m, int n, int p, int uf){

	switch(uf){
		case 16:
				return multiply_block_reorder_reuse_unroll_16(A, B, m, n, p);
				
		case 8:	
				return multiply_block_reorder_reuse_unroll_8( A, B, m, n, p);
				
  		case 4: 
		  		return multiply_block_reorder_reuse_unroll_4( A, B, m, n, p);
				
  		case 2: 
		  		return multiply_block_reorder_reuse_unroll_2( A, B, m, n, p);
				
  		default: 
		  		return multiply_block_reorder_reuse( A, B, m, n, p);
				
	}
}

// Matrix_multiplication method
 double* fast_matrix_mult(const double *a, const double *b, int m, int n, int p){
	int uf; 
	
	if(p % 16 == 0) 		uf = 16;
	else if(p % 8 == 0) 	uf = 8;
	else if(p % 4 == 0) 	uf = 4;
	else if(p % 2 == 0)		uf = 2;
	else					uf = 0;
	
	std::cout << "Matrix dimensions " << m  << "x" << n << std::endl;
	std::cout << "Block size:" << BLK_SIZE << std::endl;	
	std::cout << "Unrolls:" <<  uf << std::endl;
	
	return multiply_helper(a, b, m, n, p, uf); 				
}


// Transpose
double* matrix_transpose(const double *a, int m, int n){

	double *transpose;
	transpose = (double*)calloc(n * m, sizeof(double));

	for (int i=0; i < m; ++i)
		{
			for (int j=0; j < n; ++j)
			{
				transpose[i*n+j] = a[j*m+i];

			}
		}
	return transpose;
}


void display_matrix(const double *A, long m, long n) {

    for (long i=0; i<m; i++) {
		for (long j=0; j<n; j++)
	    	printf("%f\t", A[i*n+j]);
		printf("\n");
	}
}
