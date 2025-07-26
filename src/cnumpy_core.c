#include "cnumpy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <time.h>
#include <assert.h>

// Global state
static cnp_graph_t *g_default_graph = NULL;
static cnp_name_scope_t *g_default_scope = NULL;
static bool g_initialized = false;

// ============================================================================
// Initialization and cleanup
// ============================================================================

void cnp_init(void) {
    if (g_initialized) return;
    
    // Initialize random seed
    srand(time(NULL));
    
    // Create default graph
    g_default_graph = malloc(sizeof(cnp_graph_t));
    g_default_graph->mode = CNP_EAGER_EXECUTION;
    g_default_graph->tensor_count = 0;
    g_default_graph->capacity = 1024;
    g_default_graph->tensors = malloc(sizeof(cnp_tensor_t*) * g_default_graph->capacity);
    
    // Create default name scope
    g_default_scope = cnp_name_scope_create();
    
    g_initialized = true;
}

void cnp_cleanup(void) {
    if (!g_initialized) return;
    
    // Free all tensors in the graph
    if (g_default_graph) {
        for (size_t i = 0; i < g_default_graph->tensor_count; i++) {
            cnp_tensor_decref(g_default_graph->tensors[i]);
        }
        free(g_default_graph->tensors);
        free(g_default_graph);
        g_default_graph = NULL;
    }
    
    // Free name scope
    if (g_default_scope) {
        cnp_name_scope_free(g_default_scope);
        g_default_scope = NULL;
    }
    
    g_initialized = false;
}

// ============================================================================
// Graph management
// ============================================================================

cnp_graph_t* cnp_get_default_graph(void) {
    if (!g_initialized) cnp_init();
    return g_default_graph;
}

void cnp_set_execution_mode(cnp_execution_mode_t mode) {
    if (!g_initialized) cnp_init();
    g_default_graph->mode = mode;
}

cnp_execution_mode_t cnp_get_execution_mode(void) {
    if (!g_initialized) cnp_init();
    return g_default_graph->mode;
}

// ============================================================================
// Shape utilities
// ============================================================================

cnp_shape_t cnp_shape_create(size_t ndim, const size_t *dims) {
    cnp_shape_t shape;
    shape.ndim = ndim;
    shape.dims = malloc(sizeof(size_t) * ndim);
    shape.size = 1;
    
    for (size_t i = 0; i < ndim; i++) {
        shape.dims[i] = dims[i];
        shape.size *= dims[i];
    }
    
    return shape;
}

void cnp_shape_free(cnp_shape_t *shape) {
    if (shape->dims) {
        free(shape->dims);
        shape->dims = NULL;
    }
    shape->ndim = 0;
    shape->size = 0;
}

cnp_shape_t cnp_shape_copy(const cnp_shape_t *shape) {
    return cnp_shape_create(shape->ndim, shape->dims);
}

bool cnp_shape_equal(const cnp_shape_t *a, const cnp_shape_t *b) {
    if (a->ndim != b->ndim) return false;
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->dims[i] != b->dims[i]) return false;
    }
    return true;
}

size_t cnp_shape_size(const cnp_shape_t *shape) {
    return shape->size;
}

// ============================================================================
// Memory management
// ============================================================================

size_t cnp_dtype_size(cnp_dtype_t dtype) {
    switch (dtype) {
        case CNP_FLOAT32: return sizeof(float);
        case CNP_FLOAT64: return sizeof(double);
        case CNP_INT32: return sizeof(int32_t);
        case CNP_INT64: return sizeof(int64_t);
        default: return 0;
    }
}

cnp_tensor_t* cnp_tensor_alloc(const cnp_shape_t *shape, cnp_dtype_t dtype) {
    cnp_tensor_t *tensor = malloc(sizeof(cnp_tensor_t));
    
    // Initialize tensor
    tensor->name = NULL;
    tensor->dtype = dtype;
    tensor->shape = cnp_shape_copy(shape);
    tensor->data = malloc(cnp_dtype_size(dtype) * shape->size);
    tensor->grad = NULL;
    tensor->requires_grad = false;
    tensor->ref_count = 1;
    tensor->op = NULL;
    tensor->inputs = NULL;
    tensor->num_inputs = 0;
    
    // Add to graph
    cnp_graph_t *graph = cnp_get_default_graph();
    if (graph->tensor_count >= graph->capacity) {
        graph->capacity *= 2;
        graph->tensors = realloc(graph->tensors, sizeof(cnp_tensor_t*) * graph->capacity);
    }
    graph->tensors[graph->tensor_count++] = tensor;
    
    return tensor;
}

void cnp_tensor_incref(cnp_tensor_t *tensor) {
    if (tensor) {
        tensor->ref_count++;
    }
}

void cnp_tensor_decref(cnp_tensor_t *tensor) {
    if (!tensor) return;
    
    tensor->ref_count--;
    if (tensor->ref_count <= 0) {
        cnp_tensor_free(tensor);
    }
}

void cnp_tensor_free(cnp_tensor_t *tensor) {
    if (!tensor) return;
    
    // Free name
    if (tensor->name) {
        free(tensor->name);
    }
    
    // Free shape
    cnp_shape_free(&tensor->shape);
    
    // Free data
    if (tensor->data) {
        free(tensor->data);
    }
    
    // Free gradient
    if (tensor->grad) {
        cnp_tensor_decref(tensor->grad);
    }
    
    // Free operation
    if (tensor->op) {
        free(tensor->op);
    }
    
    // Free inputs
    if (tensor->inputs) {
        for (size_t i = 0; i < tensor->num_inputs; i++) {
            cnp_tensor_decref(tensor->inputs[i]);
        }
        free(tensor->inputs);
    }
    
    free(tensor);
}

// ============================================================================
// Tensor creation
// ============================================================================

cnp_tensor_t* cnp_tensor_create(const cnp_shape_t *shape, cnp_dtype_t dtype, const void *data) {
    cnp_tensor_t *tensor = cnp_tensor_alloc(shape, dtype);
    
    if (data) {
        memcpy(tensor->data, data, cnp_dtype_size(dtype) * shape->size);
    }
    
    return tensor;
}

cnp_tensor_t* cnp_zeros(const cnp_shape_t *shape, cnp_dtype_t dtype) {
    cnp_tensor_t *tensor = cnp_tensor_alloc(shape, dtype);
    memset(tensor->data, 0, cnp_dtype_size(dtype) * shape->size);
    return tensor;
}

cnp_tensor_t* cnp_ones(const cnp_shape_t *shape, cnp_dtype_t dtype) {
    cnp_tensor_t *tensor = cnp_tensor_alloc(shape, dtype);
    
    if (dtype == CNP_FLOAT32) {
        float *data = (float*)tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = 1.0f;
        }
    } else if (dtype == CNP_FLOAT64) {
        double *data = (double*)tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = 1.0;
        }
    } else if (dtype == CNP_INT32) {
        int32_t *data = (int32_t*)tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = 1;
        }
    } else if (dtype == CNP_INT64) {
        int64_t *data = (int64_t*)tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = 1;
        }
    }
    
    return tensor;
}

cnp_tensor_t* cnp_randn(const cnp_shape_t *shape, cnp_dtype_t dtype) {
    cnp_tensor_t *tensor = cnp_tensor_alloc(shape, dtype);
    
    if (dtype == CNP_FLOAT32) {
        float *data = (float*)tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            // Box-Muller transform for normal distribution
            static bool has_spare = false;
            static float spare;
            
            if (has_spare) {
                has_spare = false;
                data[i] = spare;
            } else {
                has_spare = true;
                float u = (float)rand() / RAND_MAX;
                float v = (float)rand() / RAND_MAX;
                float mag = sqrtf(-2.0f * logf(u));
                data[i] = mag * cosf(2.0f * M_PI * v);
                spare = mag * sinf(2.0f * M_PI * v);
            }
        }
    } else if (dtype == CNP_FLOAT64) {
        double *data = (double*)tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            static bool has_spare = false;
            static double spare;
            
            if (has_spare) {
                has_spare = false;
                data[i] = spare;
            } else {
                has_spare = true;
                double u = (double)rand() / RAND_MAX;
                double v = (double)rand() / RAND_MAX;
                double mag = sqrt(-2.0 * log(u));
                data[i] = mag * cos(2.0 * M_PI * v);
                spare = mag * sin(2.0 * M_PI * v);
            }
        }
    }
    
    return tensor;
}

cnp_tensor_t* cnp_uniform(const cnp_shape_t *shape, cnp_dtype_t dtype, float low, float high) {
    cnp_tensor_t *tensor = cnp_tensor_alloc(shape, dtype);
    
    if (dtype == CNP_FLOAT32) {
        float *data = (float*)tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = low + (high - low) * ((float)rand() / RAND_MAX);
        }
    } else if (dtype == CNP_FLOAT64) {
        double *data = (double*)tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = low + (high - low) * ((double)rand() / RAND_MAX);
        }
    }
    
    return tensor;
}

// ============================================================================
// Variable creation
// ============================================================================

cnp_var_t* cnp_var_create(const cnp_shape_t *shape, cnp_dtype_t dtype, const void *data, bool trainable) {
    cnp_var_t *var = malloc(sizeof(cnp_var_t));
    
    // Create tensor
    var->tensor = cnp_tensor_alloc(shape, dtype);
    if (data) {
        memcpy(var->tensor->data, data, cnp_dtype_size(dtype) * shape->size);
    }
    var->trainable = trainable;
    var->tensor->requires_grad = trainable;
    
    // Increment reference count since the variable now references the tensor
    cnp_tensor_incref(var->tensor);
    
    return var;
}

cnp_var_t* cnp_var_zeros(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable) {
    cnp_var_t *var = malloc(sizeof(cnp_var_t));
    
    // Create tensor
    var->tensor = cnp_tensor_alloc(shape, dtype);
    memset(var->tensor->data, 0, cnp_dtype_size(dtype) * shape->size);
    var->trainable = trainable;
    var->tensor->requires_grad = trainable;
    
    // Increment reference count since the variable now references the tensor
    cnp_tensor_incref(var->tensor);
    
    return var;
}

cnp_var_t* cnp_var_ones(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable) {
    cnp_var_t *var = malloc(sizeof(cnp_var_t));
    
    // Create tensor
    var->tensor = cnp_tensor_alloc(shape, dtype);
    
    // Set all values to 1
    if (dtype == CNP_FLOAT32) {
        float *data = (float*)var->tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = 1.0f;
        }
    } else if (dtype == CNP_FLOAT64) {
        double *data = (double*)var->tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = 1.0;
        }
    } else if (dtype == CNP_INT32) {
        int32_t *data = (int32_t*)var->tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = 1;
        }
    } else if (dtype == CNP_INT64) {
        int64_t *data = (int64_t*)var->tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = 1;
        }
    }
    
    var->trainable = trainable;
    var->tensor->requires_grad = trainable;
    
    // Increment reference count since the variable now references the tensor
    cnp_tensor_incref(var->tensor);
    
    return var;
}

cnp_var_t* cnp_var_randn(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable) {
    cnp_var_t *var = malloc(sizeof(cnp_var_t));
    
    // Create tensor
    var->tensor = cnp_tensor_alloc(shape, dtype);
    
    // Fill with random normal values
    if (dtype == CNP_FLOAT32) {
        float *data = (float*)var->tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            static bool has_spare = false;
            static float spare;
            
            if (has_spare) {
                has_spare = false;
                data[i] = spare;
            } else {
                has_spare = true;
                float u = (float)rand() / RAND_MAX;
                float v = (float)rand() / RAND_MAX;
                float mag = sqrtf(-2.0f * logf(u));
                data[i] = mag * cosf(2.0f * M_PI * v);
                spare = mag * sinf(2.0f * M_PI * v);
            }
        }
    } else if (dtype == CNP_FLOAT64) {
        double *data = (double*)var->tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            static bool has_spare = false;
            static double spare;
            
            if (has_spare) {
                has_spare = false;
                data[i] = spare;
            } else {
                has_spare = true;
                double u = (double)rand() / RAND_MAX;
                double v = (double)rand() / RAND_MAX;
                double mag = sqrt(-2.0 * log(u));
                data[i] = mag * cos(2.0 * M_PI * v);
                spare = mag * sin(2.0 * M_PI * v);
            }
        }
    }
    
    var->trainable = trainable;
    var->tensor->requires_grad = trainable;
    
    // Increment reference count since the variable now references the tensor
    cnp_tensor_incref(var->tensor);
    
    return var;
}

cnp_var_t* cnp_var_uniform(const cnp_shape_t *shape, cnp_dtype_t dtype, float low, float high, bool trainable) {
    cnp_var_t *var = malloc(sizeof(cnp_var_t));
    
    // Create tensor
    var->tensor = cnp_tensor_alloc(shape, dtype);
    
    // Fill with uniform random values
    if (dtype == CNP_FLOAT32) {
        float *data = (float*)var->tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = low + (high - low) * ((float)rand() / RAND_MAX);
        }
    } else if (dtype == CNP_FLOAT64) {
        double *data = (double*)var->tensor->data;
        for (size_t i = 0; i < shape->size; i++) {
            data[i] = low + (high - low) * ((double)rand() / RAND_MAX);
        }
    }
    
    var->trainable = trainable;
    var->tensor->requires_grad = trainable;
    
    // Increment reference count since the variable now references the tensor
    cnp_tensor_incref(var->tensor);
    
    return var;
}

void cnp_var_free(cnp_var_t *var) {
    if (var) {
        if (var->tensor) {
            cnp_tensor_decref(var->tensor);
        }
        free(var);
    }
}

// ============================================================================
// Utility functions
// ============================================================================

void* cnp_tensor_data(cnp_tensor_t *tensor) {
    return tensor->data;
}

float cnp_tensor_get_float(cnp_tensor_t *tensor, const size_t *indices) {
    // Calculate flat index
    size_t flat_idx = 0;
    size_t stride = 1;
    for (int i = tensor->shape.ndim - 1; i >= 0; i--) {
        flat_idx += indices[i] * stride;
        stride *= tensor->shape.dims[i];
    }
    
    if (tensor->dtype == CNP_FLOAT32) {
        return ((float*)tensor->data)[flat_idx];
    } else if (tensor->dtype == CNP_FLOAT64) {
        return (float)((double*)tensor->data)[flat_idx];
    } else if (tensor->dtype == CNP_INT32) {
        return (float)((int32_t*)tensor->data)[flat_idx];
    } else if (tensor->dtype == CNP_INT64) {
        return (float)((int64_t*)tensor->data)[flat_idx];
    }
    
    return 0.0f;
}

void cnp_tensor_set_float(cnp_tensor_t *tensor, const size_t *indices, float value) {
    // Calculate flat index
    size_t flat_idx = 0;
    size_t stride = 1;
    for (int i = tensor->shape.ndim - 1; i >= 0; i--) {
        flat_idx += indices[i] * stride;
        stride *= tensor->shape.dims[i];
    }
    
    if (tensor->dtype == CNP_FLOAT32) {
        ((float*)tensor->data)[flat_idx] = value;
    } else if (tensor->dtype == CNP_FLOAT64) {
        ((double*)tensor->data)[flat_idx] = (double)value;
    } else if (tensor->dtype == CNP_INT32) {
        ((int32_t*)tensor->data)[flat_idx] = (int32_t)value;
    } else if (tensor->dtype == CNP_INT64) {
        ((int64_t*)tensor->data)[flat_idx] = (int64_t)value;
    }
}

void cnp_print_shape(const cnp_shape_t *shape) {
    printf("(");
    for (size_t i = 0; i < shape->ndim; i++) {
        printf("%zu", shape->dims[i]);
        if (i < shape->ndim - 1) printf(", ");
    }
    printf(")");
}

void cnp_print_tensor(const cnp_tensor_t *tensor) {
    printf("Tensor(");
    if (tensor->name) {
        printf("name=%s, ", tensor->name);
    }
    printf("shape=");
    cnp_print_shape(&tensor->shape);
    printf(", dtype=");
    
    switch (tensor->dtype) {
        case CNP_FLOAT32: printf("float32"); break;
        case CNP_FLOAT64: printf("float64"); break;
        case CNP_INT32: printf("int32"); break;
        case CNP_INT64: printf("int64"); break;
    }
    
    printf(", data=[");
    
    // Print first few elements
    size_t max_print = tensor->shape.size > 10 ? 10 : tensor->shape.size;
    for (size_t i = 0; i < max_print; i++) {
        if (tensor->dtype == CNP_FLOAT32) {
            printf("%.6f", ((float*)tensor->data)[i]);
        } else if (tensor->dtype == CNP_FLOAT64) {
            printf("%.6f", ((double*)tensor->data)[i]);
        } else if (tensor->dtype == CNP_INT32) {
            printf("%d", ((int32_t*)tensor->data)[i]);
        } else if (tensor->dtype == CNP_INT64) {
            printf("%ld", ((int64_t*)tensor->data)[i]);
        }
        if (i < max_print - 1) printf(", ");
    }
    
    if (tensor->shape.size > 10) {
        printf("...");
    }
    
    printf("])\n");
}

// ============================================================================
// Gradient computation
// ============================================================================

void cnp_zero_grad(cnp_tensor_t *tensor) {
    if (tensor->grad) {
        cnp_tensor_decref(tensor->grad);
        tensor->grad = NULL;
    }
}

// Note: cnp_backward implementation moved to cnumpy_scope.c 