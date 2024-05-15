#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double **get_similarity_matrix(double **X, int n);
double euclidean_distance(double *point1, double *point2, int d);
double **get_diagonal_degree_matrix(double **X, int n);
double row_sum(double **A, int i, int n);

int main(int argc, char *argv[])
{

    int i, j;
    /* goal can be sym|ddg|norm */
    /* char *goal = argv[1]; */

    char *file_name = argv[1];

    FILE *text_file = fopen(file_name, "r");

    int n = 5;
    int d = 3;
    double **X;
    double **A;
    double **D;

    /* Allocate memory for vectors */
    X = (double **)malloc(n * sizeof(double *));
    if (X == NULL)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }
    for (i = 0; i < n; i++)
    {
        X[i] = (double *)malloc(d * sizeof(double));
        if (X[i] == NULL)
        {
            printf("An Error Has Occurred\n");
            for (j = 0; j < i; j++)
            {
                free(X[j]);
            }
            free(X);
            return 1;
        }
        for (j = 0; j < d; j++)
        {
            scanf("%lf,", &X[i][j]);
        }
    }

    /* print X: */
    printf("X - \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < d; j++)
        {
            printf("%.4f", X[i][j]);
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

    /* sym: Calculate and output the similarity matrix as described in 1.1 */
    A = get_similarity_matrix(X, n);

    /* print A: */
    printf("A - \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%.4f", A[i][j]);
            if (j < n - 1)
            {
                printf(",");
            }
            else
            {
                printf("\n");
            }
        }
    }

    D = get_diagonal_degree_matrix(A, n);

    /* print D: */
    printf("D - \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%.4f", D[i][j]);
            if (j < n - 1)
            {
                printf(",");
            }
            else
            {
                printf("\n");
            }
        }
    }

    return 0;
}

double **get_similarity_matrix(double **X, int n)
{
    int i, j;

    /* Allocate memory for vectors */
    double **A = (double **)malloc(n * sizeof(double *));
    if (A == NULL)
    {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    for (i = 0; i < n; i++)
    {
        A[i] = (double *)malloc(n * sizeof(double));
        if (A[i] == NULL)
        {
            printf("An Error Has Occurred\n");
            for (j = 0; j < i; j++)
            {
                free(A[j]);
            }
            free(A);
            return NULL;
        }
        for (j = 0; j < n; j++)
        {
            A[i][j] = 0;
        }
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i == j)
            {
                A[i][j] = 0;
            }
            else
            {
                A[i][j] = exp(-pow(euclidean_distance(X[i], X[j], n), 2) / 2);
            }
        }
    }

    return A;
}

double euclidean_distance(double *point1, double *point2, int d)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++)
    {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

double **get_diagonal_degree_matrix(double **A, int n)
{
    int i, j;

    /* Allocate memory for vectors */
    double **D = (double **)malloc(n * sizeof(double *));
    if (D == NULL)
    {
        printf("An Error Has Occurred\n");
        return NULL;
    }
    for (i = 0; i < n; i++)
    {
        D[i] = (double *)malloc(n * sizeof(double));
        if (D[i] == NULL)
        {
            printf("An Error Has Occurred\n");
            for (j = 0; j < i; j++)
            {
                free(D[j]);
            }
            free(D);
            return NULL;
        }
        for (j = 0; j < n; j++)
        {
            D[i][j] = 0;
        }
    }

    for (i = 0; i < n; i++)
    {
        D[i][i] = row_sum(A, i, n);
    }

    return D;
}

double row_sum(double **A, int i, int n) {
    double sum = 0.0;
    int j;
    for (j = 0; j < n; j++) {
        sum += A[i][j];
    }
    return sum;
}