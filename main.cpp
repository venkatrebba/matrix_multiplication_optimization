/**
 * @file main.cpp
 * @author Venkatarao Rebba <rebba498@gmail.com>
 * @brief : A Utility file for using lin_alg library 
 * @version 0.1
 * @date 2022-02-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>
#include <iostream>
#include <cstdlib>
#include "lin_alg.h"

using namespace std;

int main()
{
	int m, n, p;	
	double *a, *b, *c, *c2, *t1, *t2;
	static struct timeval str, end;
	unsigned long long time;
	
 	std::cout << "Please enter m: ";
    std::cin >> m;

    std::cout << "Please enter n: ";
    std::cin >> n;

    std::cout << "Please enter p: ";
    std::cin >> p;

    unsigned int seed = clock();
    srand(seed);

	a = (double*)malloc((m * n) * sizeof(double));
	b = (double*)malloc((n * p) * sizeof(double));
	c = (double*)calloc(m * p, sizeof(double));

	initialize_mat(a, m, n);
	initialize_mat(b, n, p);

	gettimeofday(&str, NULL);
	fast_matrix_mult(c, a, b, m, n, p);
	gettimeofday(&end, NULL);

	cout << "\nFast Multiplication Method" << endl;
	time = 1000 * (end.tv_sec - str.tv_sec) + (end.tv_usec - str.tv_usec) / 1000;
	cout << "Time for multiplication: " << time << "ms" << endl;

	cout << endl << "Transposing matrix-A" <<endl;
	t1 = (double*)malloc((m * n) * sizeof(double));
	gettimeofday(&str, NULL);
	fast_transpose(t1, a, m, n);
	gettimeofday(&end, NULL);

	cout << "Fast Transpose Method" << endl ;
	time = 1000 * (end.tv_sec - str.tv_sec) + (end.tv_usec - str.tv_usec) / 1000;
	cout << "Time for tranpose: " << time << "ms\n";

	free(a); free(b); free(c); free(t1);

	return 0;
}
