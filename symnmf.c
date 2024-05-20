#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void get_similarity_matrix(double **X, double **A, int n, int d);
double squared_euclidean_distance(double *point1, double *point2, int d);
void get_diagonal_degree_matrix(double **A, double **D, int n);
void get_normalized_similarity_matrix(double **A, double **D, double ***W, int n);
/* void get_clusters(double **W, double **H, int n, int k); */
double row_sum(double **A, int i, int n);
void diagonal_matrix_power(double **D, double ***inverse_root_D, int n);
void matrix_mul(double **A, double **B, int n, double ***result);
void print_matrix(double **mat, int n, int d);
void readCSV(const char *file_name, double **matrix, int rows, int cols);
void allocate_matrix(double ***mat, int n, int d);
void free_matrix(double **mat, int rows);
void countRowsAndCols(const char *file_name, int *rows, int *cols);

int main(int argc, char *argv[])
{
    char *goal, *file_name;
    double **X;
    double **A;
    double **D;
    double **W;
    int n, d;

    if (argc != 3)
        return 1;

    /* goal can be sym|ddg|norm */
    goal = argv[1];
    file_name = argv[2];

    countRowsAndCols(file_name, &n, &d);

    allocate_matrix(&X, n, d);

    readCSV(file_name, X, n, d);

    /* sym: Calculate and output the similarity matrix as described in 1.1 */
    /* Allocate memory for vectors */

    allocate_matrix(&A, n, n);
    get_similarity_matrix(X, A, n, d);

    if (strcmp(goal, "sym") == 0)
    {
        print_matrix(A, n, n);
        free_matrix(X, n);
        free_matrix(A, n);
        return 0;
    }

    allocate_matrix(&D, n, n);
    get_diagonal_degree_matrix(A, D, n);

    if (strcmp(goal, "ddg") == 0)
    {
        print_matrix(D, n, n);
        free_matrix(X, n);
        free_matrix(A, n);
        free_matrix(D, n);
        return 0;
    }

    allocate_matrix(&W, n, n);
    get_normalized_similarity_matrix(A, D, &W, n);

    free_matrix(X, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);

    if (strcmp(goal, "norm") == 0)
    {
        print_matrix(W, n, n);
        free_matrix(X, n);
        free_matrix(A, n);
        free_matrix(D, n);
        free_matrix(W, n);
        return 0;
    }

    return 0;
}

void countRowsAndCols(const char *file_name, int *rows, int *cols)
{
    FILE *file = fopen(file_name, "r");
    char ch;
    int line_cols = 0;
    *rows = 0;
    *cols = 0;

    if (file == NULL)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    while ((ch = fgetc(file)) != EOF)
    {
        if (ch == ',')
        {
            line_cols++;
        }
        else if (ch == '\n')
        {
            (*rows)++;
            line_cols++;
            if (line_cols > *cols)
            {
                *cols = line_cols;
            }
            line_cols = 0;
        }
    }

    fclose(file);
}

void readCSV(const char *file_name, double **matrix, int rows, int cols)
{
    int i, j;
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            if (fscanf(file, "%lf,", &matrix[i][j]) != 1)
            {
                perror("Error reading file");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
}

void get_similarity_matrix(double **X, double **A, int n, int d)
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
                double squared_distance = squared_euclidean_distance(X[i], X[j], d);
                double value = exp(-squared_distance / 2);
                A[i][j] = A[j][i] = value;
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
        sum += difference * difference;
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

void diagonal_matrix_power(double **D, double ***inverse_root_D, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        if (D[i][i] == 0)
            (*inverse_root_D)[i][i] = 0;
        else
            (*inverse_root_D)[i][i] = 1 / sqrt(D[i][i]);
    }
}

void matrix_mul(double **A, double **B, int n, double ***result)
{
    int i, j, k;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            (*result)[i][j] = 0;

            for (k = 0; k < n; k++)
            {
                (*result)[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void get_normalized_similarity_matrix(double **A, double **D, double ***W, int n)
{
    double **inverse_root_D;
    double **tmp;
    allocate_matrix(&inverse_root_D, n, n);
    allocate_matrix(&tmp, n, n);

    diagonal_matrix_power(D, &inverse_root_D, n);

    matrix_mul(inverse_root_D, A, n, &tmp);
    matrix_mul(tmp, inverse_root_D, n, W);
}

/* void get_clusters(double **W, double **H, int n, int k)
// {
//     int difference = 1;
//     int iter = 1;

//     double **WH, **HTH, **HTH_H, **numerator, **denominator;
//     allocate_matrix(&WH, n, k);
//     allocate_matrix(&HTH, k, k);
//     allocate_matrix(&HTH_H, n, k);
//     allocate_matrix(&numerator, n, k);
//     allocate_matrix(&denominator, n, k);

//     matrix_mul_not_square(W, H, n, n, k, WH);

//     double **H_transpose;
//     allocate_matrix(&H_transpose, k, n);
//     transpose(H, &H_transpose, n, k);

//     matrix_mul_not_square(H_transpose, H, n, k, k, HTH);
//     matrix_mul_not_square(H, HTH, n, k, k, HTH_H);

//     elementwise_divide();

//     while (iter <= 300)
//     {
//         double norm = frobenius_norm(H_new, H, n, k);
//         if (norm < 0.0001)
//         {
//             break;
//         }
//         H = H_new;

//         iter++;
//     }
// }

// void update_matrix(double **H_new, double **H, double **W, int rows, int cols, int common_dim, double beta)
// {
//     double **WH, **HTH, **HTH_H, **numerator, **denominator;

//     // Allocate temporary matrices
//     allocate_matrix(&WH, rows, cols);
//     allocate_matrix(&HTH, cols, cols);
//     allocate_matrix(&HTH_H, rows, cols);
//     allocate_matrix(&numerator, rows, cols);
//     allocate_matrix(&denominator, rows, cols);

//     // WH = W * H
//     matrix_multiply(WH, W, H, rows, common_dim, cols);

//     // HTH = H^T * H
//     double **H_transpose;
//     allocate_matrix(&H_transpose, cols, common_dim);
//     for (int i = 0; i < common_dim; i++)
//     {
//         for (int j = 0; j < cols; j++)
//         {
//             H_transpose[j][i] = H[i][j];
//         }
//     }
//     matrix_multiply(HTH, H_transpose, H, cols, common_dim, cols);

//     // HTH_H = H * HTH
//     matrix_multiply(HTH_H, H, HTH, rows, cols, cols);

//     // Numerator: WH
//     // Denominator: HTH_H
//     elementwise_divide(numerator, WH, HTH_H, rows, cols);

//     // H_new = H * (1 - beta + beta * numerator)
//     for (int i = 0; i < rows; i++)
//     {
//         for (int j = 0; j < cols; j++)
//         {
//             H_new[i][j] = H[i][j] * (1 - beta + beta * numerator[i][j]);
//         }
//     }

//     // Free temporary matrices
//     free_matrix(WH, rows);
//     free_matrix(HTH, cols);
//     free_matrix(HTH_H, rows);
//     free_matrix(numerator, rows);
//     free_matrix(denominator, rows);
//     free_matrix(H_transpose, cols);
// }

// double frobenius_norm(double **mat1, double **mat2, int rows, int cols)
// {
//     double sum = 0.0;
//     for (int i = 0; i < rows; i++)
//     {
//         for (int j = 0; j < cols; j++)
//         {
//             double diff = mat1[i][j] - mat2[i][j];
//             sum += diff * diff;
//         }
//     }
//     return sqrt(sum);
// }

// void transpose(double **mat, double ***result, int rows, int cols)
// {
//     int i, j;

//     for (i = 0; i < rows; i++)
//     {
//         for (j = 0; j < cols; j++)
//         {
//             (*result)[i][j] = mat[j][i];
//         }
//     }
// }

// void substruct_matrices(double **A, double **B, int n, double ***result)
// {
//     int i, j;

//     for (i = 0; i < n; i++)
//     {
//         for (j = 0; j < n; j++)
//         {
//             (*result)[i][j] = A[i][j] - B[i][j];
//         }
//     }
// }

// void elementwise_divide(double **A, double **B, int rows, int cols, double ***result)
// {
//     int i, j;

//     for (i = 0; i < rows; i++)
//     {
//         for (j = 0; j < cols; j++)
//         {
//             if (B[i][j] != 0)
//             {
//                 (*result)[i][j] = A[i][j] / B[i][j];
//             }
//             else
//             {
//                 (*result)[i][j] = 0;
//             }
//         }
//     }
// }

// void elementwise_multiply(double **result, double **mat1, double **mat2, int rows, int cols)
// {
//     for (int i = 0; i < rows; i++)
//     {
//         for (int j = 0; j < cols; j++)
//         {
//             result[i][j] = mat1[i][j] * mat2[i][j];
//         }
//     }
// }

*/

void allocate_matrix(double ***matrix, int rows, int cols)
{
    int i;
    *matrix = (double **)malloc(rows * sizeof(double *));
    for (i = 0; i < rows; i++)
    {
        (*matrix)[i] = (double *)malloc(cols * sizeof(double));
    }
}

void free_matrix(double **mat, int rows)
{
    int i;
    for (i = 0; i < rows; i++)
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
