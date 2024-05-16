#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

static PyObject *sym(PyObject *self, PyObject *args)
{

    /* Declare variables */
    int n, d, i, j;
    PyObject *vectors;
    double **X;
    double **A;
    PyObject *final_result;
    PyObject *py_value;
    double num;

    /* Parse Python arguments */
    if (!PyArg_ParseTuple(args, "iiO", &n, &d, &vectors))
    {
        return NULL;
    }

    /* Allocate memory for X - the input points */
    X = (double **)malloc(n * sizeof(double *));
    if (X == NULL)
    {
        printf("An Error Has Occurred\n");
        return NULL;
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
            return NULL;
        }
    }

    /* Allocate memory for A - the similarity matrix */
    A = (double **)malloc(n * sizeof(double *));
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
    }

    // convert python floats to c doubles
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < d; j++)
        {
            py_value = PyList_GetItem(PyList_GetItem(vectors, i), j);
            num = PyFloat_AsDouble(py_value);
            X[i][j] = num;
        }
    }

    A = get_similarity_matrix(X, n);

    /* Parse A from c doubles to python floats */
    final_result = PyTuple_New(n);
    for (i = 0; i < n; i++)
    {
        PyObject *vector_tuple = PyTuple_New(n);
        for (j = 0; j < n; j++)
        {
            PyObject *float_obj = Py_BuildValue("d", A[i][j]);
            PyTuple_SetItem(vector_tuple, j, float_obj);
        }
        PyTuple_SetItem(final_result, i, vector_tuple);
    }

    return Py_BuildValue("O", final_result);
}

static PyObject *ddg(PyObject *self, PyObject *args)
{

    /* Declare variables */
    int n, d, i, j;
    PyObject *vectors;
    double **X;
    double **A;
    double **D;
    PyObject *final_result;
    PyObject *py_value;
    double num;

    /* Parse Python arguments */
    if (!PyArg_ParseTuple(args, "iiO", &n, &d, &vectors))
    {
        return NULL;
    }

    /* Allocate memory for X - the input points */
    X = (double **)malloc(n * sizeof(double *));
    if (X == NULL)
    {
        printf("An Error Has Occurred\n");
        return NULL;
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
            return NULL;
        }
    }

    /* Allocate memory for A - the similarity matrix */
    A = (double **)malloc(n * sizeof(double *));
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
    }

    /* Allocate memory for D - the diagonal degree matrix */
    D = (double **)malloc(n * sizeof(double *));
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
    }

    // convert python floats to c doubles
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < d; j++)
        {
            py_value = PyList_GetItem(PyList_GetItem(vectors, i), j);
            num = PyFloat_AsDouble(py_value);
            X[i][j] = num;
        }
    }

    A = get_similarity_matrix(X, n);
    D = get_diagonal_degree_matrix(A, n);

    /* Parse A from c doubles to python floats */
    final_result = PyTuple_New(n);
    for (i = 0; i < n; i++)
    {
        PyObject *vector_tuple = PyTuple_New(n);
        for (j = 0; j < n; j++)
        {
            PyObject *float_obj = Py_BuildValue("d", D[i][j]);
            PyTuple_SetItem(vector_tuple, j, float_obj);
        }
        PyTuple_SetItem(final_result, i, vector_tuple);
    }

    return Py_BuildValue("O", final_result);
}

static PyObject *norm(PyObject *self, PyObject *args)
{

    /* Declare variables */
    int n, d, i, j;
    PyObject *vectors;
    double **X;
    double **A;
    double **D;
    double **W;
    PyObject *final_result;
    PyObject *py_value;
    double num;

    /* Parse Python arguments */
    if (!PyArg_ParseTuple(args, "iiO", &n, &d, &vectors))
    {
        return NULL;
    }

    /* Allocate memory for X - the input points */
    X = (double **)malloc(n * sizeof(double *));
    if (X == NULL)
    {
        printf("An Error Has Occurred\n");
        return NULL;
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
            return NULL;
        }
    }

    /* Allocate memory for A - the similarity matrix */
    A = (double **)malloc(n * sizeof(double *));
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
    }

    /* Allocate memory for D - the diagonal degree matrix */
    D = (double **)malloc(n * sizeof(double *));
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
    }

    /* Allocate memory for W - the input points */
    W = (double **)malloc(n * sizeof(double *));
    if (W == NULL)
    {
        printf("An Error Has Occurred\n");
        return NULL;
    }

    for (i = 0; i < n; i++)
    {
        W[i] = (double *)malloc(n * sizeof(double));
        if (W[i] == NULL)
        {
            printf("An Error Has Occurred\n");
            for (j = 0; j < i; j++)
            {
                free(W[j]);
            }
            free(W);
            return NULL;
        }
    }

    // convert python floats to c doubles
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < d; j++)
        {
            py_value = PyList_GetItem(PyList_GetItem(vectors, i), j);
            num = PyFloat_AsDouble(py_value);
            X[i][j] = num;
        }
    }

    A = get_similarity_matrix(X, n);
    D = get_diagonal_degree_matrix(A, n);
    W = get_normalized_similarity_matrix(A, D, n);

    /* Parse A from c doubles to python floats */
    final_result = PyTuple_New(n);
    for (i = 0; i < n; i++)
    {
        PyObject *vector_tuple = PyTuple_New(n);
        for (j = 0; j < n; j++)
        {
            PyObject *float_obj = Py_BuildValue("d", W[i][j]);
            PyTuple_SetItem(vector_tuple, j, float_obj);
        }
        PyTuple_SetItem(final_result, i, vector_tuple);
    }

    return Py_BuildValue("O", final_result);
}

static PyMethodDef symnmfMethods[] = {
    {"sym",                                                                       /* the Python method name that will be used */
     (PyCFunction)sym,                                                            /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                                                                /* flags indicating parameters
                                                                                     accepted for this function */
     PyDoc_STR("Calculate similarity matrix from a given matrix. \n Arguments:\n" /* The docstring for the function */
               "n = number of vectors given.\n"
               "d = number of dimensions of the vectors.\n"
               "vectors - a list of list contains the vectors to be clustered")},
    {"ddg",                                                                            /* the Python method name that will be used */
     (PyCFunction)ddg,                                                                 /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                                                                     /* flags indicating parameters
                                                                                          accepted for this function */
     PyDoc_STR("Calculate diagonal degree matrix from a given matrix. \n Arguments:\n" /* The docstring for the function */
               "n = number of vectors given.\n"
               "d = number of dimensions of the vectors.\n"
               "vectors - a list of list contains the vectors to be clustered")},
    {"norm",                                                                                 /* the Python method name that will be used */
     (PyCFunction)norm,                                                                      /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                                                                           /* flags indicating parameters
                                                                                                accepted for this function */
     PyDoc_STR("Calculate normalized similarity matrix from a given matrix. \n Arguments:\n" /* The docstring for the function */
               "n = number of vectors given.\n"
               "d = number of dimensions of the vectors.\n"
               "vectors - a list of list contains the vectors to be clustered")},
    {NULL, NULL, 0, NULL} /* The last entry must be all NULL as shown to act as a
                             sentinel. Python looks for this entry to know that all
                             of the functions for the module have been defined. */
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf_capi", /* name of module */
    NULL,          /* module documentation, may be NULL */
    -1,            /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    symnmfMethods  /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_symnmf_capi(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m)
    {
        return NULL;
    }
    return m;
}