#ifndef CAP_H
#define CAP_H

void get_similarity_matrix(double **X, double **A, int n);
void get_diagonal_degree_matrix(double **A, double **D, int n);
void get_normalized_similarity_matrix(double **A, double **D, double ***W, int n);

double squared_euclidean_distance(double *point1, double *point2, int d);
double row_sum(double **A, int i, int n);
void allocate_matrix(double **mat, int n, int d);
void print_matrix(double **mat, int n, int d);

#endif