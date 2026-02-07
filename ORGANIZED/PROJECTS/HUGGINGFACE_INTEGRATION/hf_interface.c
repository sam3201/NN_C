#include "hf_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>

// Python module for Hugging Face interface
static PyObject* hf_module = NULL;
static PyObject* hf_init_func = NULL;
static PyObject* hf_free_func = NULL;
static PyObject* hf_embeddings_func = NULL;
static PyObject* hf_generate_func = NULL;
static PyObject* hf_dim_func = NULL;

// Initialize Python interface
static int init_python_interface(void) {
    if (hf_module != NULL) return 1;  // Already initialized
    
    Py_Initialize();
    if (!Py_IsInitialized()) {
        fprintf(stderr, "Failed to initialize Python\n");
        return 0;
    }
    
    // Add current directory to Python path
    PyObject* sys_path = PySys_GetObject("path");
    char current_dir[1024];
    if (getcwd(current_dir, sizeof(current_dir)) != NULL) {
        PyObject* path = PyUnicode_FromString(current_dir);
        PyList_Append(sys_path, path);
        Py_DECREF(path);
    }
    
    // Import the hf_interface module
    hf_module = PyImport_ImportModule("hf_interface");
    if (!hf_module) {
        PyErr_Print();
        fprintf(stderr, "Failed to import hf_interface module\n");
        Py_Finalize();
        return 0;
    }
    
    // Get function references
    hf_init_func = PyObject_GetAttrString(hf_module, "init_model");
    hf_free_func = PyObject_GetAttrString(hf_module, "free_model");
    hf_embeddings_func = PyObject_GetAttrString(hf_module, "_c_get_embeddings");
    hf_generate_func = PyObject_GetAttrString(hf_module, "_c_generate_text");
    hf_dim_func = PyObject_GetAttrString(hf_module, "get_model_dim");
    
    if (!hf_init_func || !hf_free_func || !hf_embeddings_func || 
        !hf_generate_func || !hf_dim_func) {
        PyErr_Print();
        fprintf(stderr, "Failed to get function references\n");
        Py_Finalize();
        return 0;
    }
    
    return 1;
}

HF_Model HF_init_model(const char* model_name) {
    if (!init_python_interface()) {
        return NULL;
    }
    
    PyObject* args = Py_BuildValue("(s)", model_name);
    if (!args) {
        PyErr_Print();
        return NULL;
    }
    
    PyObject* result = PyObject_CallObject(hf_init_func, args);
    Py_DECREF(args);
    
    if (!result) {
        PyErr_Print();
        return NULL;
    }
    
    long model_id = PyLong_AsLong(result);
    Py_DECREF(result);
    
    if (model_id == 0) {
        return NULL;
    }
    
    // Store as pointer (we'll use the Python ID)
    return (HF_Model)(intptr_t)model_id;
}

void HF_free_model(HF_Model model) {
    if (!model || !hf_free_func) return;
    
    long model_id = (long)(intptr_t)model;
    PyObject* args = Py_BuildValue("(l)", model_id);
    if (args) {
        PyObject_CallObject(hf_free_func, args);
        Py_DECREF(args);
    }
}

int HF_get_embeddings(HF_Model model, const char* text, long double* embeddings, size_t model_dim) {
    if (!model || !text || !embeddings || !hf_embeddings_func) {
        return 0;
    }
    
    long model_id = (long)(intptr_t)model;
    
    // Create numpy array for embeddings
    PyObject* embeddings_array = PyArray_SimpleNew(1, (npy_intp*)&model_dim, NPY_DOUBLE);
    if (!embeddings_array) {
        PyErr_Print();
        return 0;
    }
    
    PyObject* args = Py_BuildValue("(lsO)", model_id, text, embeddings_array, (int)model_dim);
    if (!args) {
        Py_DECREF(embeddings_array);
        PyErr_Print();
        return 0;
    }
    
    PyObject* result = PyObject_CallObject(hf_embeddings_func, args);
    Py_DECREF(args);
    Py_DECREF(embeddings_array);
    
    if (!result) {
        PyErr_Print();
        return 0;
    }
    
    int success = PyLong_AsLong(result);
    Py_DECREF(result);
    
    if (success) {
        // Copy from numpy array to C array
        double* data = (double*)PyArray_DATA(embeddings_array);
        for (size_t i = 0; i < model_dim; i++) {
            embeddings[i] = (long double)data[i];
        }
    }
    
    return success;
}

int HF_generate_text(HF_Model model, const char* prompt, size_t max_length, char* output, size_t output_size) {
    if (!model || !prompt || !output || !hf_generate_func) {
        return 0;
    }
    
    long model_id = (long)(intptr_t)model;
    
    PyObject* args = Py_BuildValue("(lsI)", model_id, prompt, (unsigned int)max_length);
    if (!args) {
        PyErr_Print();
        return 0;
    }
    
    PyObject* result = PyObject_CallObject(hf_generate_func, args);
    Py_DECREF(args);
    
    if (!result) {
        PyErr_Print();
        return 0;
    }
    
    const char* generated = PyBytes_AsString(result);
    if (generated) {
        strncpy(output, generated, output_size - 1);
        output[output_size - 1] = '\0';
    }
    
    Py_DECREF(result);
    return 1;
}

size_t HF_get_model_dim(HF_Model model) {
    if (!model || !hf_dim_func) {
        return 0;
    }
    
    long model_id = (long)(intptr_t)model;
    
    PyObject* args = Py_BuildValue("(l)", model_id);
    if (!args) {
        return 0;
    }
    
    PyObject* result = PyObject_CallObject(hf_dim_func, args);
    Py_DECREF(args);
    
    if (!result) {
        PyErr_Print();
        return 0;
    }
    
    size_t dim = (size_t)PyLong_AsLong(result);
    Py_DECREF(result);
    
    return dim;
}

// Simplified version using subprocess for better compatibility
// This version uses a simpler approach with JSON communication

