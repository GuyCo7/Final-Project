#ifndef CAP_H
#define CAP_H

double **get_similarity_matrix(double **X, int n);
double **get_diagonal_degree_matrix(double **A, int n);
double **get_normalized_similarity_matrix(double **A, double **D, int n);
double euclidean_distance(double *point1, double *point2, int d);

#endif