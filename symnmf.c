#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BETA 0.5
#define EPSILON 0.0001
#define MAX_ITER 300

void get_similarity_matrix(double **X, double **A, int n, int d);
double squared_euclidean_distance(double *point1, double *point2, int d);
void get_diagonal_degree_matrix(double **A, double **D, int n);
void get_normalized_similarity_matrix(double **A, double **D, double ***W, int n);
void get_clusters(double **W, double **H, double ***next_H, int n, int k);
double row_sum(double **A, int i, int n);
void get_diagonal_inverse_root(double **D, double ***inverse_root_D, int n);
void print_matrix(double **mat, int n, int d);
void readCSV(const char *file_name, double **matrix, int rows, int cols);
void allocate_matrix(double ***mat, int n, int d);
void free_matrix(double **mat, int rows);
void countRowsAndCols(const char *file_name, int *rows, int *cols);
void multiply_matrices(double **A, double **B, double ***C, int n, int m, int k);
double frobenius_norm(double **A, double **B, int rows, int cols);
void transpose(double **mat, double ***result, int rows, int cols);
void copy_matrix(double **A, double ***B, int n, int k);

int main(int argc, char *argv[])
{
    char *goal, *file_name;
    double **X;
    double **A;
    double **D;
    double **W;
    int n, d;

    if (argc != 3)
    {
        printf("An Error Has Occurred");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];

    countRowsAndCols(file_name, &n, &d);

    allocate_matrix(&X, n, d);

    readCSV(file_name, X, n, d);

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

    if (strcmp(goal, "norm") == 0)
    {
        print_matrix(W, n, n);
    }

    free_matrix(X, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);

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
        perror("An Error Has Occurred");
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
        perror("An Error Has Occurred");
        exit(1);
    }

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            if (fscanf(file, "%lf,", &matrix[i][j]) != 1)
            {
                perror("An Error Has Occurred");
                fclose(file);
                exit(1);
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

void get_diagonal_inverse_root(double **D, double ***inverse_root_D, int n)
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

void get_normalized_similarity_matrix(double **A, double **D, double ***W, int n)
{
    double **inverse_root_D;
    double **IR_D_mul_A;

    allocate_matrix(&inverse_root_D, n, n);
    allocate_matrix(&IR_D_mul_A, n, n);

    get_diagonal_inverse_root(D, &inverse_root_D, n);

    multiply_matrices(inverse_root_D, A, &IR_D_mul_A, n, n, n);
    multiply_matrices(IR_D_mul_A, inverse_root_D, W, n, n, n);

    free_matrix(inverse_root_D, n);
    free_matrix(IR_D_mul_A, n);
}

void copy_matrix(double **source, double ***destination, int n, int k)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            (*destination)[i][j] = source[i][j];
        }
    }
}

void get_clusters(double **W, double **H, double ***next_H, int n, int k)
{
    int i, j;
    int iter = 0;

    double **H_transpose, **WH, **HTH, **HTH_H;
    allocate_matrix(&WH, n, k);
    allocate_matrix(&HTH, n, n);
    allocate_matrix(&HTH_H, n, k);
    allocate_matrix(&H_transpose, k, n);

    while (iter <= MAX_ITER)
    {
        double norm;

        if (iter != 0)
        {
            copy_matrix(*next_H, &H, n, k);
        }

        multiply_matrices(W, H, &WH, n, n, k);

        transpose(H, &H_transpose, n, k);
        multiply_matrices(H, H_transpose, &HTH, n, k, n);
        multiply_matrices(HTH, H, &HTH_H, n, n, k);

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < k; j++)
            {
                (*next_H)[i][j] = H[i][j] * ((1 - BETA) + (BETA * (WH[i][j] / HTH_H[i][j])));
            }
        }

        norm = frobenius_norm(*next_H, H, n, k);
        if (norm < EPSILON)
        {
            break;
        }

        iter++;
    }

    free_matrix(H, n);
    free_matrix(WH, n);
    free_matrix(HTH, n);
    free_matrix(HTH_H, n);
    free_matrix(H_transpose, k);
}

double frobenius_norm(double **A, double **B, int rows, int cols)
{
    int i, j;
    double sum = 0.0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            double diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

void transpose(double **mat, double ***result, int rows, int cols)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            (*result)[j][i] = mat[i][j];
        }
    }
}

void multiply_matrices(double **A, double **B, double ***C, int rowsA, int colsA, int colsB)
{
    int i, j, k;
    for (i = 0; i < rowsA; i++)
    {
        for (j = 0; j < colsB; j++)
        {
            (*C)[i][j] = 0.0;
            for (k = 0; k < colsA; k++)
            {
                (*C)[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void allocate_matrix(double ***matrix, int rows, int cols)
{
    int i;
    *matrix = (double **)calloc(rows, sizeof(double *));
    for (i = 0; i < rows; i++)
    {
        (*matrix)[i] = (double *)calloc(cols, sizeof(double));
    }
}

void free_matrix(double **mat, int rows)
{
    int i;
    if (mat == NULL)
    {
        printf("An Error Has Occurred");
        return;
    }
    for (i = 0; i < rows; i++)
    {
        if (mat[i] != NULL)
        {
            free(mat[i]);
            mat[i] = NULL;
        }
    }
    free(mat);
    mat = NULL;
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
        }
        printf("\n");
        fflush(stdout);
    }

    return;
}
