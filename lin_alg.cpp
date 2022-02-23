/**
 * @file lin_alg.cpp
 * @author Venkatarao Rebba <rebba498@gmail.com>
 * @brief a Linear Algebra library for matrix operations
 * @version 0.1
 * @date 2022-02-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include "lin_alg.h"

#define BLK_SIZE 128
#define min(a,b) (((a)<(b))?(a):(b))
#define THRESHOLD 0.001

//Function for initializing matrix with random values
void initialize_mat(double* A, int m, int n){
	int i, j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)	
			A[i*n+j] = rand() % 10;
}


// Naive Matrix multiplication method
void naive_multiply(double* C, const double *A, const double *B, int m, int n, int p){

    int i, j, k;

    for(i = 0; i < m; i++)
        for(j = 0; j < p; j++)
			for(k = 0; k < n; k++)
                C[p*i+j] += A[n*i+k] * B[p*k+j];				      
}


// Block Multiplication
void multiply_block(double* C, const double *A, const double *B, int m, int n, int p){ 

	int i, j, k, ii, jj, kk, bs = BLK_SIZE;
	for(ii = 0; ii < m; ii += bs)
		for(jj = 0; jj < p; jj += bs)
			for(kk = 0; kk < n; kk += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(j = jj; j < min(p, jj+bs); j++)
						for(k = kk; k < min(n, kk+bs); k++)
							C[p*i+j] += A[n*i+k] * B[p*k+j];							
}

// Multiplication with blocking and Loop reorder
void multiply_block_reorder(double* C, const double *A, const double *B, int m, int n, int p){ 
	
    int i, j, k, ii, jj, kk, bs = BLK_SIZE;
	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
						for(j = jj; j < min(p, jj+bs); j++)
							C[p*i+j] += A[n*i+k] * B[p*k+j];						
				
}

// Multiplication with blocking, Loop reorder and reuse
void multiply_block_reorder_reuse(double* C, const double *A, const double *B, int m, int n, int p){ 
	
    int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;
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
	
}

// Multiplication with blocking, Loop reorder, reuse and unrolling
void multiply_block_reorder_reuse_unroll_2(double* C, const double *A, const double *B, int m, int n, int p){
	
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;
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
	
}

// Multiplication with blocking, Loop reorder, reuse and unrolling
void multiply_block_reorder_reuse_unroll_4(double *C, const double *A, const double *B, int m, int n, int p){
	
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;
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
	
}

// Multiplication with blocking, Loop reorder, reuse and unrolling
void multiply_block_reorder_reuse_unroll_8( double *C,  const double *A, const double *B, int m, int n, int p){ 

	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;
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
	
}

// Multiplication with blocking, Loop reorder, reuse and unrolling
void multiply_block_reorder_reuse_unroll_16(double *C,   const double *A, const double *B, int m, int n, int p){
    
	int i, j, k, ii, jj, kk, Aik, bs = BLK_SIZE;
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
}

// Multiplciation helper
void multiply_helper(double *C, const double *A, const double *B, int m, int n, int p, int uf){

	switch(uf){
		case 16:
				return multiply_block_reorder_reuse_unroll_16(C, A, B, m, n, p);
				
		case 8:	
				return multiply_block_reorder_reuse_unroll_8(C, A, B, m, n, p);
				
  		case 4: 
		  		return multiply_block_reorder_reuse_unroll_4(C, A, B, m, n, p);
				
  		case 2: 
		  		return multiply_block_reorder_reuse_unroll_2(C, A, B, m, n, p);
				
  		default: 
		  		return multiply_block_reorder_reuse(C, A, B, m, n, p);
				
	}
}

// Matrix_multiplication method
void fast_matrix_mult(double* c, const double *a, const double *b, int m, int n, int p){

	int uf; 
	
	if(p % 16 == 0) 		uf = 16;
	else if(p % 8 == 0) 	uf = 8;
	else if(p % 4 == 0) 	uf = 4;
	else if(p % 2 == 0)		uf = 2;
	else					uf = 0;
	
	// std::cout << "Matrix dimensions " << m  << "x" << n << std::endl;
	// std::cout << "Block size:" << BLK_SIZE << std::endl;	
	// std::cout << "Unrolls:" <<  uf << std::endl;
	
	return multiply_helper(c, a, b, m, n, p, uf); 				
}


// Naive Transpose
void matrix_transpose(double *transpose, const double *a, int m, int n){

	for (int i=0; i < m; ++i)
		{
			for (int j=0; j < n; ++j)
			{
				transpose[j*m+i] = a[i*n+j];
				//destination[j+i*n] = source[i+j*n];
			}
		}
	
}

// Fast Transpose
void fast_transpose(double *dst, const double *src, int n, int p) {
    
    size_t block = 32;
    for (size_t i = 0; i < n; i += block) {
        for(size_t j = 0; j < p; ++j) {
            for(size_t b = 0; b < block && i + b < n; ++b) {
                dst[j*n + i + b] = src[(i + b)*p + j];
            }
        }
    }
}

// To print the matrix
void display_matrix(const double *A, int m, int n) {
    for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++)
            std::cout << A[i*n+j] << " ";
		std::cout << std::endl;
	}
}

// To compare two matrices
bool is_two_matrices_equal(int m, int n, const double *mat1, const double *mat2){
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++) {
			int idx = i*n+j;
			if (abs(mat1[idx] - mat2[idx]) > THRESHOLD) {
				printf("ERROR! Elements at %d,%d %f != %f\n", i, j, mat1[idx], mat2[idx]);
				return 0;
			}
		}
	printf("OK.\n");
	return 1;
}