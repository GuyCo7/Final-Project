#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

static PyObject *sym(PyObject *self, PyObject *args)
{
    int n, d, i, j;
    PyObject *vectors;
    double **X;
    double **A;
    PyObject *final_result;
    PyObject *py_value;
    double num;

    if (!PyArg_ParseTuple(args, "iiO", &n, &d, &vectors))
    {
        return NULL;
    }

    allocate_matrix(&X, n, d);

    allocate_matrix(&A, n, n);

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

    get_similarity_matrix(X, A, n);

    /* Parse A from c doubles to python floats */
    final_result = PyList_New(n);
    for (i = 0; i < n; i++)
    {
        PyObject *row_list = PyList_New(n);
        for (j = 0; j < n; j++)
        {
            PyObject *float_obj = PyFloat_FromDouble(A[i][j]);
            PyList_SET_ITEM(row_list, j, float_obj);
        }
        PyList_SET_ITEM(final_result, i, row_list);
    }

    free(X);
    free(A);

    return final_result;
}

static PyObject *ddg(PyObject *self, PyObject *args)
{
    int n, i, j;
    PyObject *vectors;
    double **A;
    double **D;
    PyObject *final_result;
    PyObject *py_value;
    double num;

    if (!PyArg_ParseTuple(args, "iO", &n, &vectors))
    {
        return NULL;
    }

    allocate_matrix(&A, n, n);

    allocate_matrix(&D, n, n);

    // convert python floats to c doubles
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            py_value = PyList_GetItem(PyList_GetItem(vectors, i), j);
            num = PyFloat_AsDouble(py_value);
            A[i][j] = num;
        }
    }

    get_diagonal_degree_matrix(A, D, n);

    /* Parse A from c doubles to python floats */
    final_result = PyList_New(n);
    for (i = 0; i < n; i++)
    {
        PyObject *row_list = PyList_New(n);
        for (j = 0; j < n; j++)
        {
            PyObject *float_obj = PyFloat_FromDouble(D[i][j]);
            PyList_SET_ITEM(row_list, j, float_obj);
        }
        PyList_SET_ITEM(final_result, i, row_list);
    }

    free(A);
    free(D);

    return final_result;
}

static PyObject *norm(PyObject *self, PyObject *args)
{
    int n, i, j;
    PyObject *py_A;
    PyObject *py_D;
    double **A;
    double **D;
    double **W;
    PyObject *final_result;
    PyObject *py_value;
    double num;

    if (!PyArg_ParseTuple(args, "iOO", &n, &py_A, &py_D))
    {
        return NULL;
    }

    allocate_matrix(&A, n, n);

    allocate_matrix(&D, n, n);

    allocate_matrix(&W, n, n);

    // convert python floats to c doubles
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            py_value = PyList_GetItem(PyList_GetItem(py_A, i), j);
            num = PyFloat_AsDouble(py_value);
            A[i][j] = num;
        }
    }

    // convert python floats to c doubles
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            py_value = PyList_GetItem(PyList_GetItem(py_D, i), j);
            num = PyFloat_AsDouble(py_value);
            D[i][j] = num;
        }
    }

    get_normalized_similarity_matrix(A, D, W, n);
    print_matrix(W, n, n);

    /* Parse A from c doubles to python floats */
    final_result = PyList_New(n);
    for (i = 0; i < n; i++)
    {
        PyObject *row_list = PyList_New(n);
        for (j = 0; j < n; j++)
        {
            PyObject *float_obj = PyFloat_FromDouble(W[i][j]);
            PyList_SET_ITEM(row_list, j, float_obj);
        }
        PyList_SET_ITEM(final_result, i, row_list);
    }

    free(A);
    free(D);
    free(W);

    return final_result;
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