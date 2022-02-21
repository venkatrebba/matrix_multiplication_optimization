#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>
#include <iostream>
#include <cstdlib>
#include "matrix_operations.h"


using namespace std;

int main()
{
	int m, n, p;	
	double *a, *b, *c, *c2, *c3;
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


	bool out;
	
	a = (double*)malloc((m * n) * sizeof(double));
	b = (double*)malloc((n * p) * sizeof(double));
	c = (double*)calloc(m * p, sizeof(double));

	
	initialize_mat(a, m, n);
	initialize_mat(b, n, p);


	c2 = (double*)calloc(m * p, sizeof(double));

	gettimeofday(&str, NULL);
	double* d = fast_matrix_mult(a, b, m, n, p);
	gettimeofday(&end, NULL);

	cout << "\nNaive Method\n" ;
	time = 1000 * (end.tv_sec - str.tv_sec) + (end.tv_usec - str.tv_usec) / 1000;
	cout << "Time: " << time << "ms\n";


	display_matrix(a, m, m);
	double* f = matrix_transpose(a, m, n);

	printf("AFte transpose");
	display_matrix(f, m, m);

	free(a); free(b);
	return 0;
}
