#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void get_similarity_matrix(double **X, double **A, int n);
double squared_euclidean_distance(double *point1, double *point2, int d);
void get_diagonal_degree_matrix(double **A, double **D, int n);
void get_normalized_similarity_matrix(double **A, double **D, double **W, int n);
void get_clusters(double **W, double **H, int n, int k);
double row_sum(double **A, int i, int n);
void diagonal_matrix_power(double **D, double **inverse_root_D, double p, int n);
void matrix_mul(double **A, double **B, int n, double ***result);
void print_matrix(double **mat, int n, int d);
void allocate_matrix(double **mat, int n, int d);

int main(int argc, char *argv[])
{
    int n = 5;
    int d = 5;
    double **X;
    double **A;
    double **D;
    double **W;

    /* goal can be sym|ddg|norm */
    char *goal = argv[1];

    /* Allocate memory for vectors */
    allocate_matrix(X, n, d);
    printf("X - \n");
    print_matrix(X, n, d);

    /* sym: Calculate and output the similarity matrix as described in 1.1 */
    /* Allocate memory for vectors */
    allocate_matrix(A, n, n);
    get_similarity_matrix(X, A, n);

    /* print A: */
    printf("A - \n");
    print_matrix(A, n, n);

    allocate_matrix(D, n, n);
    get_diagonal_degree_matrix(A, D, n);

    /* print D: */
    printf("D - \n");
    print_matrix(D, n, n);

    allocate_matrix(W, n, n);
    get_normalized_similarity_matrix(A, D, W, n);

    /* print W: */
    printf("W - \n");
    print_matrix(W, n, n);

    return 0;
}

void get_similarity_matrix(double **X, double **A, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = i; j < n; j++)
        {
            if (i == j)
            {
                A[i][j] = 0;
            }
            else
            {
                double squared_distance = squared_euclidean_distance(X[i], X[j], n);
                A[i][j] = A[j][i] = exp(-squared_distance / 2);
            }
        }
    }
}

double squared_euclidean_distance(double *point1, double *point2, int d)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++)
    {
        double difference = point1[i] - point2[i];
        sum += difference * difference; // difference^2
    }
    return sum;
}

void get_diagonal_degree_matrix(double **A, double **D, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        D[i][i] = row_sum(A, i, n);
    }
}

double row_sum(double **A, int i, int n)
{
    double sum = 0.0;
    int j;
    for (j = 0; j < n; j++)
    {
        sum += A[i][j];
    }
    return sum;
}

void diagonal_matrix_power(double **D, double ***inverse_root_D, double q, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        (*inverse_root_D)[i][i] = 1 / sqrt(D[i][i]);
    }
}

void matrix_mul(double **A, double **B, int n, double ***result)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            (*result)[i][j] = 0;

            for (int k = 0; k < n; k++)
            {
                (*result)[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_multiply(double **result, double **mat1, double **mat2, int rows1, int cols1, int cols2)
{
    for (int i = 0; i < rows1; i++)
    {
        for (int j = 0; j < cols2; j++)
        {
            result[i][j] = 0.0;
            for (int k = 0; k < cols1; k++)
            {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

void get_normalized_similarity_matrix(double **A, double **D, double ***W, int n)
{
    double **inverse_root_D;
    allocate_matrix(&inverse_root_D, n, n);
    double **tmp;
    allocate_matrix(&tmp, n, n);

    diagonal_matrix_power(D, &inverse_root_D, -(1 / 2), n);

    matrix_mul(inverse_root_D, A, n, &tmp);
    matrix_mul(tmp, inverse_root_D, n, &W);
}

void get_clusters(double **W, double **H, int n, int k)
{
    int difference = 1;
    int iter = 1;

    double **WH, **HTH, **HTH_H, **numerator, **denominator;
    allocate_matrix(&WH, n, k);
    allocate_matrix(&HTH, k, k);
    allocate_matrix(&HTH_H, n, k);
    allocate_matrix(&numerator, n, k);
    allocate_matrix(&denominator, n, k);

    matrix_mul_not_square(W, H, n, n, k, WH);

    double **H_transpose;
    allocate_matrix(&H_transpose, k, n);
    transpose(H, &H_transpose, n, k);

    matrix_mul_not_square(H_transpose, H, n, k, k, HTH);
    matrix_mul_not_square(H, HTH, n, k, k, HTH_H);

    elementwise_divide();

    while (iter <= 300)
    {
        double norm = frobenius_norm(H_new, H, n, k);
        if (norm < 0.0001)
        {
            break;
        }
        H = H_new;

        iter++;
    }
}

void update_matrix(double **H_new, double **H, double **W, int rows, int cols, int common_dim, double beta)
{
    double **WH, **HTH, **HTH_H, **numerator, **denominator;

    // Allocate temporary matrices
    allocate_matrix(&WH, rows, cols);
    allocate_matrix(&HTH, cols, cols);
    allocate_matrix(&HTH_H, rows, cols);
    allocate_matrix(&numerator, rows, cols);
    allocate_matrix(&denominator, rows, cols);

    // WH = W * H
    matrix_multiply(WH, W, H, rows, common_dim, cols);

    // HTH = H^T * H
    double **H_transpose;
    allocate_matrix(&H_transpose, cols, common_dim);
    for (int i = 0; i < common_dim; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            H_transpose[j][i] = H[i][j];
        }
    }
    matrix_multiply(HTH, H_transpose, H, cols, common_dim, cols);

    // HTH_H = H * HTH
    matrix_multiply(HTH_H, H, HTH, rows, cols, cols);

    // Numerator: WH
    // Denominator: HTH_H
    elementwise_divide(numerator, WH, HTH_H, rows, cols);

    // H_new = H * (1 - beta + beta * numerator)
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            H_new[i][j] = H[i][j] * (1 - beta + beta * numerator[i][j]);
        }
    }

    // Free temporary matrices
    free_matrix(WH, rows);
    free_matrix(HTH, cols);
    free_matrix(HTH_H, rows);
    free_matrix(numerator, rows);
    free_matrix(denominator, rows);
    free_matrix(H_transpose, cols);
}

double frobenius_norm(double **mat1, double **mat2, int rows, int cols)
{
    double sum = 0.0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double diff = mat1[i][j] - mat2[i][j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}

void transpose(double **mat, double ***result, int rows, int cols)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            (*result)[i][j] = mat[j][i];
        }
    }
}

void substruct_matrices(double **A, double **B, int n, double ***result)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            (*result)[i][j] = A[i][j] - B[i][j];
        }
    }
}

void elementwise_divide(double **A, double **B, int rows, int cols, double ***result)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            if (B[i][j] != 0)
            {
                (*result)[i][j] = A[i][j] / B[i][j];
            }
            else
            {
                (*result)[i][j] = 0;
            }
        }
    }
}

void elementwise_multiply(double **result, double **mat1, double **mat2, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i][j] = mat1[i][j] * mat2[i][j];
        }
    }
}

void allocate_matrix(double ***mat, int n, int d)
{
    int i, j;

    *mat = (double **)malloc(n * sizeof(double *));
    if (*mat == NULL)
    {
        printf("An Error Has Occurred\n");
        return;
    }
    for (i = 0; i < n; i++)
    {
        (*mat)[i] = (double *)malloc(d * sizeof(double));
        if ((*mat)[i] == NULL)
        {
            printf("An Error Has Occurred\n");
            for (j = 0; j < i; j++)
            {
                free((*mat)[j]);
            }
            free((*mat));
            return;
        }
        for (j = 0; j < d; j++)
        {
            (*mat)[i][j] = 0;
        }
    }

    return;
}

void free_matrix(double **mat, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(mat[i]);
    }
    free(mat);
}

void print_matrix(double **mat, int n, int d)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < d; j++)
        {
            printf("%.4f", mat[i][j]);
            if (j < d - 1)
            {
                printf(",");
            }
            else
            {
                printf("\n");
            }
        }
    }

    return;
}