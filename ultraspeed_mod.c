//
// Created by dhonchar on 23 Oct 2024.
//
#ifndef FAST_ERROR_H
#define FAST_ERROR_H

#endif //FAST_ERROR_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
// Include your C function declarations
#include "ultraspeed.h"

//void matmul_optimized_packed(float* A, float* C, int n);
//void matmul_optimized_bitwise_simd(float* A, float* C, int n);


static PyObject* matmul_ssyrk_py(PyObject* self, PyObject* args) {
    PyArrayObject *A_obj, *C_obj;
    int dim;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &A_obj, &PyArray_Type, &C_obj, &dim)) {
        return NULL;
    }

    // Ensure that A and C are NumPy arrays of type float32
    if (PyArray_TYPE(A_obj) != NPY_FLOAT32 || PyArray_TYPE(C_obj) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "A and C must be NumPy arrays of type float32.");
        return NULL;
    }

    // Ensure that A and C are two-dimensional
    if (PyArray_NDIM(A_obj) != 2 || PyArray_NDIM(C_obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and C must be two-dimensional arrays.");
        return NULL;
    }

    // Ensure that the dimensions match
    npy_intp* A_dims = PyArray_DIMS(A_obj);
    npy_intp* C_dims = PyArray_DIMS(C_obj);
    if (A_dims[0] != dim || A_dims[1] != dim || C_dims[0] != dim || C_dims[1] != dim) {
        PyErr_SetString(PyExc_ValueError, "Dimensions of A and C must match 'dim'.");
        return NULL;
    }

    // Get pointers to the data as C-style arrays
    float* A = (float*)PyArray_DATA(A_obj);
    float* C = (float*)PyArray_DATA(C_obj);

    // Call your C functions
    matmul_blis_ssyrk(A, C, dim, dim);
//    matmul_optimized_packed_simd(A,C, dim);
    return PyLong_FromLong(0);
}
//void matmul_blis_gemm(float* A, float* B, float* C, int m, int n, int k);

static PyObject* matmul_gemm_py(PyObject* self, PyObject* args) {
    PyArrayObject *A_obj, *B_obj, *C_obj;
    int dim;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &A_obj, &PyArray_Type, &B_obj, &PyArray_Type, &C_obj, &dim)) {
        return NULL;
    }

    // Ensure that A and C are NumPy arrays of type float32
    if (PyArray_TYPE(A_obj) != NPY_FLOAT32 || PyArray_TYPE(C_obj) != NPY_FLOAT32 || PyArray_TYPE(B_obj) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "A, B, C must be NumPy arrays of type float32.");
        return NULL;
    }

    // Ensure that A and C are two-dimensional
    if (PyArray_NDIM(A_obj) != 2 || PyArray_NDIM(C_obj) != 2 || PyArray_NDIM(B_obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and C must be two-dimensional arrays.");
        return NULL;
    }

    // Ensure that the dimensions match
    npy_intp* A_dims = PyArray_DIMS(A_obj);
    npy_intp* C_dims = PyArray_DIMS(C_obj);
    if (A_dims[0] != dim || A_dims[1] != dim || C_dims[0] != dim || C_dims[1] != dim) {
        PyErr_SetString(PyExc_ValueError, "Dimensions of A and C must match 'dim'.");
        return NULL;
    }

    // Get pointers to the data as C-style arrays
    float* A = (float*)PyArray_DATA(A_obj);
    float* B = (float*)PyArray_DATA(B_obj);
    float* C = (float*)PyArray_DATA(C_obj);

    // Call your C functions
    matmul_blis_gemm(A, B, C, dim, dim, dim);
    //    matmul_optimized_packed_simd(A,C, dim);
    return PyLong_FromLong(0);
}


//static PyObject* matmul_naive_py(PyObject* self, PyObject* args) {
//    PyArrayObject *A_obj, *B_obj, *C_obj;
//    int dim;
//
//    // Parse the input tuple
//    if (!PyArg_ParseTuple(args, "O!O!O!i", &PyArray_Type, &A_obj, &PyArray_Type, &B_obj, &PyArray_Type, &C_obj, &dim)) {
//        return NULL;
//    }
//
//    // Ensure that A and C are NumPy arrays of type float32
//    if (PyArray_TYPE(A_obj) != NPY_FLOAT32 || PyArray_TYPE(C_obj) != NPY_FLOAT32 | | PyArray_TYPE(B_obj) != NPY_FLOAT32) {
//        PyErr_SetString(PyExc_TypeError, "A, B, C must be NumPy arrays of type float32.");
//        return NULL;
//    }
//
//    // Ensure that A and C are two-dimensional
//    if (PyArray_NDIM(A_obj) != 2 || PyArray_NDIM(C_obj) != 2 || PyArray_NDIM(B_obj) != 2) {
//        PyErr_SetString(PyExc_ValueError, "A and C must be two-dimensional arrays.");
//        return NULL;
//    }
//
//    // Ensure that the dimensions match
//    npy_intp* A_dims = PyArray_DIMS(A_obj);
//    npy_intp* C_dims = PyArray_DIMS(C_obj);
//    if (A_dims[0] != dim || A_dims[1] != dim || C_dims[0] != dim || C_dims[1] != dim) {
//        PyErr_SetString(PyExc_ValueError, "Dimensions of A and C must match 'dim'.");
//        return NULL;
//    }
//
//    // Get pointers to the data as C-style arrays
//    float* A = (float*)PyArray_DATA(A_obj);
//    float* B = (float*)PyArray_DATA(B_obj);
//    float* C = (float*)PyArray_DATA(C_obj);
//
//    // Call your C functions
//    matmul_naive_binary(A, B, C, dim);
//    //    matmul_optimized_packed_simd(A,C, dim);
//}


static PyObject* matmul_matmul_binary_py(PyObject* self, PyObject* args) {
    PyArrayObject *A_obj, *C_obj;
    int dim;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &A_obj, &PyArray_Type, &C_obj, &dim)) {
        return NULL;
    }

    // Ensure that A and C are NumPy arrays of type float32
    if (PyArray_TYPE(A_obj) != NPY_FLOAT32 || PyArray_TYPE(C_obj) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "A and C must be NumPy arrays of type float32.");
        return NULL;
    }

    // Ensure that A and C are two-dimensional
    if (PyArray_NDIM(A_obj) != 2 || PyArray_NDIM(C_obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and C must be two-dimensional arrays.");
        return NULL;
    }

    // Ensure that the dimensions match
    npy_intp* A_dims = PyArray_DIMS(A_obj);
    npy_intp* C_dims = PyArray_DIMS(C_obj);
    if (A_dims[0] != dim || A_dims[1] != dim || C_dims[0] != dim || C_dims[1] != dim) {
        PyErr_SetString(PyExc_ValueError, "Dimensions of A and C must match 'dim'.");
        return NULL;
    }

    // Get pointers to the data as C-style arrays
    float* A = (float*)PyArray_DATA(A_obj);
    float* C = (float*)PyArray_DATA(C_obj);

    // Call your C functions
    matmul_optimized_packed(A, C, dim);
    //    matmul_optimized_packed_simd(A,C, dim);
    return PyLong_FromLong(0);
}




static PyObject* matmul_matmul_bool_py(PyObject* self, PyObject* args) {
    PyArrayObject *A_obj, *C_obj;
    int dim;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &A_obj, &PyArray_Type, &C_obj, &dim)) {
        return NULL;
    }

    // Ensure that A and C are NumPy arrays of type float32
    if (PyArray_TYPE(A_obj) != NPY_BOOL || PyArray_TYPE(C_obj) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "A and C must be NumPy arrays of type bool and float32.");
        return NULL;
    }

    // Ensure that A and C are two-dimensional
    if (PyArray_NDIM(A_obj) != 2 || PyArray_NDIM(C_obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and C must be two-dimensional arrays.");
        return NULL;
    }

    // Ensure that the dimensions match
    npy_intp* A_dims = PyArray_DIMS(A_obj);
    npy_intp* C_dims = PyArray_DIMS(C_obj);
    if (A_dims[0] != dim || A_dims[1] != dim || C_dims[0] != dim || C_dims[1] != dim) {
        PyErr_SetString(PyExc_ValueError, "Dimensions of A and C must match 'dim'.");
        return NULL;
    }

    // Get pointers to the data as C-style arrays
    char* A = (char*)PyArray_DATA(A_obj);
    float* C = (float*)PyArray_DATA(C_obj);

    // Call your C functions
    matmul_optimized_packed_bool(A, C, dim);
    //    matmul_optimized_packed_simd(A,C, dim);
    return PyLong_FromLong(0);
}

static PyObject* matmul_matmul_binary_simd_py(PyObject* self, PyObject* args) {
    PyArrayObject *A_obj, *C_obj;
    int dim;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &A_obj, &PyArray_Type, &C_obj, &dim)) {
        return NULL;
    }

    // Ensure that A and C are NumPy arrays of type float32
    if (PyArray_TYPE(A_obj) != NPY_FLOAT32 || PyArray_TYPE(C_obj) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "A and C must be NumPy arrays of type float32.");
        return NULL;
    }

    // Ensure that A and C are two-dimensional
    if (PyArray_NDIM(A_obj) != 2 || PyArray_NDIM(C_obj) != 2) {
        PyErr_SetString(PyExc_ValueError, "A and C must be two-dimensional arrays.");
        return NULL;
    }

    // Ensure that the dimensions match
    npy_intp* A_dims = PyArray_DIMS(A_obj);
    npy_intp* C_dims = PyArray_DIMS(C_obj);
    if (A_dims[0] != dim || A_dims[1] != dim || C_dims[0] != dim || C_dims[1] != dim) {
        PyErr_SetString(PyExc_ValueError, "Dimensions of A and C must match 'dim'.");
        return NULL;
    }

    // Get pointers to the data as C-style arrays
    float* A = (float*)PyArray_DATA(A_obj);
    float* C = (float*)PyArray_DATA(C_obj);

    // Call your C functions
    matmul_optimized_bitwise_simd(A, C, dim);
    //    matmul_optimized_packed_simd(A,C, dim);
    return PyLong_FromLong(0);
}



static PyMethodDef MyMethods[] = {
    {"matmul_ssyrk", matmul_ssyrk_py, METH_VARARGS, "Matrix multiplication optimized C functions (SSYRK)."},
    {"matmul_gemm", matmul_gemm_py, METH_VARARGS, "Matrix multiplication optimized C functions (GEMM)."},
//    {"matmul_naive", matmul_naive_py, METH_VARARGS, "Matrix multiplication, naive C implementation"},
    {"matmul_bitwise", matmul_matmul_binary_py, METH_VARARGS, "Matrix multiplication optimized C functions - bitwise."},
    {"matmul_bitwise_bool", matmul_matmul_bool_py, METH_VARARGS, "Matrix multiplication optimized C functions - bool bitwise."},
    {"matmul_bitwise_simd", matmul_matmul_binary_simd_py, METH_VARARGS, "Matrix multiplication optimized C functions - bitwise (simd)."},

    {NULL, NULL, 0, NULL}  // Sentinel
};

static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "ultraspeed",           // Module name
    NULL,                  // Module documentation
    -1,                    // Size of per-interpreter state of the module
    MyMethods
};

PyMODINIT_FUNC PyInit_ultraspeed(void) {
    import_array();  // Initialize NumPy API
    return PyModule_Create(&mymodule);
}


