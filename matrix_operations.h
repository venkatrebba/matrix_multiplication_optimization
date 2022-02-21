#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 

//Function for initializing matrix with random values
void initialize_mat(double* A, int m, int n);

// Naive Matrix multiplication method
double* naive_multiply(const double *A, const double *B, int m, int n, int p);

// Matrix_multiplication method
double* fast_matrix_mult(const double *a, const double *b, int m, int n, int p);

// Transpose
double* matrix_transpose(const double *a, int m, int n);

// Display matrix
void display_matrix(const double *A, long m, long n);

#endif /* MATRIX_OPERATIONS_H_ */
