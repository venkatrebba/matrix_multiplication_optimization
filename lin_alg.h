/**
 * @file lin_alg.h
 * @author Venkatarao Rebba <rebba498@gmail.com>
 * @brief  Header file for linear algebra library
 * @version 0.1
 * @date 2022-02-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 

//Function for initializing matrix with random values
void initialize_mat(double* A, int rows, int cols);

// Naive Matrix multiplication method
// void naive_multiply(double *C, const double *A, const double *B, int m, int n, int p);

// Matrix_multiplication method
void fast_matrix_mult(double *C, const double *a, const double *b, int m, int n, int p);

// Naive Transpose
// void matrix_transpose(double *dst, const double *a, int m, int n);

// Fast Transpose
void fast_transpose(double *dst, const double *src, int rows, int cols);

// Display matrix
void display_matrix(const double *A, int rows, int cols);

// Tow compare two matrices
bool is_two_matrices_equal(const double *mat1, const double *mat2, int rows, int cols); 

#endif /* MATRIX_OPERATIONS_H_ */
