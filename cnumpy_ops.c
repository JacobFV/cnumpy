#include "cnumpy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ============================================================================
// Operation parameter structures
// ============================================================================

typedef struct {
    int axis;
} cnp_reduce_params_t;

typedef struct {
    float power;
} cnp_pow_params_t;

typedef struct {
    int *axes;
    size_t num_axes;
} cnp_transpose_params_t;

// Helper function to create or get gradient tensor
static cnp_tensor_t* cnp_get_or_create_grad(cnp_tensor_t *tensor) {
    if (!tensor->grad) {
        tensor->grad = cnp_zeros(&tensor->shape, tensor->dtype);
        cnp_tensor_incref(tensor->grad);  // Increment reference count
    }
    return tensor->grad;
}

typedef struct {
    cnp_shape_t new_shape;
} cnp_reshape_params_t;

typedef struct {
    int axis;
    size_t *orig_shapes;
    size_t num_tensors;
} cnp_concat_params_t;

// ============================================================================
// Helper functions
// ============================================================================

static cnp_tensor_t* cnp_create_op_tensor(cnp_op_type_t op_type, cnp_tensor_t **inputs, size_t num_inputs, 
                                         cnp_forward_fn_t forward_fn, cnp_backward_fn_t backward_fn, 
                                         void *params, const cnp_shape_t *output_shape) {
    cnp_tensor_t *result = cnp_tensor_alloc(output_shape, inputs[0]->dtype);
    
    // Set up operation
    result->op = malloc(sizeof(cnp_op_t));
    result->op->type = op_type;
    result->op->forward = forward_fn;
    result->op->backward = backward_fn;
    result->op->params = params;
    
    // Set up inputs
    result->inputs = malloc(sizeof(cnp_tensor_t*) * num_inputs);
    result->num_inputs = num_inputs;
    for (size_t i = 0; i < num_inputs; i++) {
        result->inputs[i] = inputs[i];
        cnp_tensor_incref(inputs[i]);
    }
    
    // Set requires_grad if any input requires grad
    for (size_t i = 0; i < num_inputs; i++) {
        if (inputs[i]->requires_grad) {
            result->requires_grad = true;
            break;
        }
    }
    
    // Execute forward pass in eager mode
    if (cnp_get_execution_mode() == CNP_EAGER_EXECUTION) {
        cnp_tensor_t *forward_result = forward_fn(inputs, num_inputs, params);
        memcpy(result->data, forward_result->data, 
               cnp_dtype_size(result->dtype) * result->shape.size);
        cnp_tensor_decref(forward_result);
    }
    
    return result;
}

// ============================================================================
// Basic arithmetic operations
// ============================================================================

static cnp_tensor_t* cnp_add_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *b = inputs[1];
    
    // For now, assume same shape (broadcasting not implemented)
    assert(cnp_shape_equal(&a->shape, &b->shape));
    
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *b_data = (float*)b->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = a_data[i] + b_data[i];
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *b_data = (double*)b->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = a_data[i] + b_data[i];
        }
    }
    
    return result;
}

static void cnp_add_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *b = inputs[1];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                a_grad[i] += out_grad[i];
            }
        }
    }
    
    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = cnp_zeros(&b->shape, b->dtype);
        }
        // grad_b += grad_output
        if (b->dtype == CNP_FLOAT32) {
            float *b_grad = (float*)b->grad->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < b->shape.size; i++) {
                b_grad[i] += out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_add(cnp_tensor_t *a, cnp_tensor_t *b) {
    cnp_tensor_t *inputs[] = {a, b};
    return cnp_create_op_tensor(CNP_OP_ADD, inputs, 2, cnp_add_forward, cnp_add_backward, NULL, &a->shape);
}

static cnp_tensor_t* cnp_sub_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *b = inputs[1];
    
    assert(cnp_shape_equal(&a->shape, &b->shape));
    
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *b_data = (float*)b->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = a_data[i] - b_data[i];
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *b_data = (double*)b->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = a_data[i] - b_data[i];
        }
    }
    
    return result;
}

static void cnp_sub_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *b = inputs[1];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                a_grad[i] += out_grad[i];
            }
        }
    }
    
    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = cnp_zeros(&b->shape, b->dtype);
        }
        // grad_b -= grad_output
        if (b->dtype == CNP_FLOAT32) {
            float *b_grad = (float*)b->grad->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < b->shape.size; i++) {
                b_grad[i] -= out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_sub(cnp_tensor_t *a, cnp_tensor_t *b) {
    cnp_tensor_t *inputs[] = {a, b};
    return cnp_create_op_tensor(CNP_OP_SUB, inputs, 2, cnp_sub_forward, cnp_sub_backward, NULL, &a->shape);
}

static cnp_tensor_t* cnp_mul_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *b = inputs[1];
    
    assert(cnp_shape_equal(&a->shape, &b->shape));
    
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *b_data = (float*)b->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = a_data[i] * b_data[i];
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *b_data = (double*)b->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = a_data[i] * b_data[i];
        }
    }
    
    return result;
}

static void cnp_mul_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *b = inputs[1];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += b * grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *b_data = (float*)b->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                a_grad[i] += b_data[i] * out_grad[i];
            }
        }
    }
    
    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = cnp_zeros(&b->shape, b->dtype);
        }
        // grad_b += a * grad_output
        if (b->dtype == CNP_FLOAT32) {
            float *b_grad = (float*)b->grad->data;
            float *a_data = (float*)a->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < b->shape.size; i++) {
                b_grad[i] += a_data[i] * out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_mul(cnp_tensor_t *a, cnp_tensor_t *b) {
    cnp_tensor_t *inputs[] = {a, b};
    return cnp_create_op_tensor(CNP_OP_MUL, inputs, 2, cnp_mul_forward, cnp_mul_backward, NULL, &a->shape);
}

static cnp_tensor_t* cnp_neg_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = -a_data[i];
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = -a_data[i];
        }
    }
    
    return result;
}

static void cnp_neg_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a -= grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                a_grad[i] -= out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_neg(cnp_tensor_t *a) {
    cnp_tensor_t *inputs[] = {a};
    return cnp_create_op_tensor(CNP_OP_NEG, inputs, 1, cnp_neg_forward, cnp_neg_backward, NULL, &a->shape);
}

// ============================================================================
// Matrix operations
// ============================================================================

static cnp_tensor_t* cnp_matmul_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *b = inputs[1];
    
    // For now, handle 2D matrices only
    assert(a->shape.ndim == 2 && b->shape.ndim == 2);
    assert(a->shape.dims[1] == b->shape.dims[0]);
    
    size_t result_dims[] = {a->shape.dims[0], b->shape.dims[1]};
    cnp_shape_t result_shape = cnp_shape_create(2, result_dims);
    cnp_tensor_t *result = cnp_tensor_alloc(&result_shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *b_data = (float*)b->data;
        float *result_data = (float*)result->data;
        
        size_t m = a->shape.dims[0];
        size_t n = a->shape.dims[1];
        size_t p = b->shape.dims[1];
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < p; j++) {
                result_data[i * p + j] = 0.0f;
                for (size_t k = 0; k < n; k++) {
                    result_data[i * p + j] += a_data[i * n + k] * b_data[k * p + j];
                }
            }
        }
    }
    
    cnp_shape_free(&result_shape);
    return result;
}

static void cnp_matmul_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *b = inputs[1];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += grad_output @ b.T
        // This is a simplified implementation
        // Full implementation would need proper matrix multiplication
    }
    
    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = cnp_zeros(&b->shape, b->dtype);
        }
        // grad_b += a.T @ grad_output
        // This is a simplified implementation
    }
}

cnp_tensor_t* cnp_matmul(cnp_tensor_t *a, cnp_tensor_t *b) {
    cnp_tensor_t *inputs[] = {a, b};
    size_t result_dims[] = {a->shape.dims[0], b->shape.dims[1]};
    cnp_shape_t result_shape = cnp_shape_create(2, result_dims);
    cnp_tensor_t *result = cnp_create_op_tensor(CNP_OP_MATMUL, inputs, 2, cnp_matmul_forward, cnp_matmul_backward, NULL, &result_shape);
    cnp_shape_free(&result_shape);
    return result;
}

// ============================================================================
// Activation functions
// ============================================================================

static cnp_tensor_t* cnp_relu_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = a_data[i] > 0.0f ? a_data[i] : 0.0f;
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = a_data[i] > 0.0 ? a_data[i] : 0.0;
        }
    }
    
    return result;
}

static void cnp_relu_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += (a > 0) * grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *a_data = (float*)a->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                a_grad[i] += (a_data[i] > 0.0f ? 1.0f : 0.0f) * out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_relu(cnp_tensor_t *a) {
    cnp_tensor_t *inputs[] = {a};
    return cnp_create_op_tensor(CNP_OP_RELU, inputs, 1, cnp_relu_forward, cnp_relu_backward, NULL, &a->shape);
}

static cnp_tensor_t* cnp_sigmoid_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = 1.0f / (1.0f + expf(-a_data[i]));
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = 1.0 / (1.0 + exp(-a_data[i]));
        }
    }
    
    return result;
}

static void cnp_sigmoid_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += sigmoid(a) * (1 - sigmoid(a)) * grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *output_data = (float*)output->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                float sig = output_data[i];
                a_grad[i] += sig * (1.0f - sig) * out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_sigmoid(cnp_tensor_t *a) {
    cnp_tensor_t *inputs[] = {a};
    return cnp_create_op_tensor(CNP_OP_SIGMOID, inputs, 1, cnp_sigmoid_forward, cnp_sigmoid_backward, NULL, &a->shape);
}

static cnp_tensor_t* cnp_tanh_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = tanhf(a_data[i]);
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = tanh(a_data[i]);
        }
    }
    
    return result;
}

static void cnp_tanh_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += (1 - tanh^2(a)) * grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *output_data = (float*)output->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                float tanh_val = output_data[i];
                a_grad[i] += (1.0f - tanh_val * tanh_val) * out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_tanh(cnp_tensor_t *a) {
    cnp_tensor_t *inputs[] = {a};
    return cnp_create_op_tensor(CNP_OP_TANH, inputs, 1, cnp_tanh_forward, cnp_tanh_backward, NULL, &a->shape);
}

static cnp_tensor_t* cnp_exp_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = expf(a_data[i]);
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = exp(a_data[i]);
        }
    }
    
    return result;
}

static void cnp_exp_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += exp(a) * grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *output_data = (float*)output->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                a_grad[i] += output_data[i] * out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_exp(cnp_tensor_t *a) {
    cnp_tensor_t *inputs[] = {a};
    return cnp_create_op_tensor(CNP_OP_EXP, inputs, 1, cnp_exp_forward, cnp_exp_backward, NULL, &a->shape);
}

static cnp_tensor_t* cnp_linear_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    // Linear is just identity: y = x
    memcpy(result->data, a->data, cnp_dtype_size(a->dtype) * a->shape.size);
    
    return result;
}

static void cnp_linear_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += grad_output (identity gradient)
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                a_grad[i] += out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_linear(cnp_tensor_t *a) {
    cnp_tensor_t *inputs[] = {a};
    return cnp_create_op_tensor(CNP_OP_LINEAR, inputs, 1, cnp_linear_forward, cnp_linear_backward, NULL, &a->shape);
}

// ============================================================================
// Power operation
// ============================================================================

static cnp_tensor_t* cnp_pow_forward(cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_pow_params_t *pow_params = (cnp_pow_params_t*)params;
    cnp_tensor_t *result = cnp_tensor_alloc(&a->shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = powf(a_data[i], pow_params->power);
        }
    } else if (a->dtype == CNP_FLOAT64) {
        double *a_data = (double*)a->data;
        double *result_data = (double*)result->data;
        
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[i] = pow(a_data[i], pow_params->power);
        }
    }
    
    return result;
}

static void cnp_pow_backward(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params) {
    cnp_tensor_t *a = inputs[0];
    cnp_pow_params_t *pow_params = (cnp_pow_params_t*)params;
    
    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = cnp_zeros(&a->shape, a->dtype);
        }
        // grad_a += power * a^(power-1) * grad_output
        if (a->dtype == CNP_FLOAT32) {
            float *a_grad = (float*)a->grad->data;
            float *a_data = (float*)a->data;
            float *out_grad = (float*)output->grad->data;
            for (size_t i = 0; i < a->shape.size; i++) {
                float grad_coeff = pow_params->power * powf(a_data[i], pow_params->power - 1.0f);
                a_grad[i] += grad_coeff * out_grad[i];
            }
        }
    }
}

cnp_tensor_t* cnp_pow(cnp_tensor_t *a, float power) {
    cnp_tensor_t *inputs[] = {a};
    cnp_pow_params_t *params = malloc(sizeof(cnp_pow_params_t));
    params->power = power;
    return cnp_create_op_tensor(CNP_OP_POW, inputs, 1, cnp_pow_forward, cnp_pow_backward, params, &a->shape);
}

// ============================================================================
// Reduction operations (simplified implementations)
// ============================================================================

cnp_tensor_t* cnp_reduce_sum(cnp_tensor_t *a, int axis) {
    // Simplified implementation - sum all elements to scalar
    size_t scalar_dims[] = {1};
    cnp_shape_t scalar_shape = cnp_shape_create(1, scalar_dims);
    cnp_tensor_t *result = cnp_tensor_alloc(&scalar_shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        result_data[0] = 0.0f;
        for (size_t i = 0; i < a->shape.size; i++) {
            result_data[0] += a_data[i];
        }
    }
    
    cnp_shape_free(&scalar_shape);
    return result;
}

cnp_tensor_t* cnp_reduce_max(cnp_tensor_t *a, int axis) {
    // Simplified implementation - max of all elements
    size_t scalar_dims[] = {1};
    cnp_shape_t scalar_shape = cnp_shape_create(1, scalar_dims);
    cnp_tensor_t *result = cnp_tensor_alloc(&scalar_shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        result_data[0] = a_data[0];
        for (size_t i = 1; i < a->shape.size; i++) {
            if (a_data[i] > result_data[0]) {
                result_data[0] = a_data[i];
            }
        }
    }
    
    cnp_shape_free(&scalar_shape);
    return result;
}

cnp_tensor_t* cnp_reduce_min(cnp_tensor_t *a, int axis) {
    // Simplified implementation - min of all elements
    size_t scalar_dims[] = {1};
    cnp_shape_t scalar_shape = cnp_shape_create(1, scalar_dims);
    cnp_tensor_t *result = cnp_tensor_alloc(&scalar_shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        result_data[0] = a_data[0];
        for (size_t i = 1; i < a->shape.size; i++) {
            if (a_data[i] < result_data[0]) {
                result_data[0] = a_data[i];
            }
        }
    }
    
    cnp_shape_free(&scalar_shape);
    return result;
}

// ============================================================================
// Shape operations (simplified implementations)
// ============================================================================

cnp_tensor_t* cnp_transpose(cnp_tensor_t *a, const int *axes) {
    // Simplified implementation - only supports 2D transpose
    if (a->shape.ndim != 2) {
        return NULL;  // Only 2D transpose supported for now
    }
    
    size_t transposed_dims[] = {a->shape.dims[1], a->shape.dims[0]};
    cnp_shape_t transposed_shape = cnp_shape_create(2, transposed_dims);
    cnp_tensor_t *result = cnp_tensor_alloc(&transposed_shape, a->dtype);
    
    if (a->dtype == CNP_FLOAT32) {
        float *a_data = (float*)a->data;
        float *result_data = (float*)result->data;
        
        size_t rows = a->shape.dims[0];
        size_t cols = a->shape.dims[1];
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result_data[j * rows + i] = a_data[i * cols + j];
            }
        }
    }
    
    cnp_shape_free(&transposed_shape);
    return result;
}

cnp_tensor_t* cnp_reshape(cnp_tensor_t *a, const cnp_shape_t *new_shape) {
    // Check that total size matches
    if (a->shape.size != new_shape->size) {
        return NULL;
    }
    
    cnp_tensor_t *result = cnp_tensor_alloc(new_shape, a->dtype);
    
    // Copy data (reshape is just a view change)
    memcpy(result->data, a->data, cnp_dtype_size(a->dtype) * a->shape.size);
    
    return result;
}

cnp_tensor_t* cnp_concat(cnp_tensor_t **tensors, size_t num_tensors, int axis) {
    // Simplified implementation - concatenate along axis 0 only
    if (num_tensors == 0) return NULL;
    
    // Calculate new shape
    size_t new_size = 0;
    for (size_t i = 0; i < num_tensors; i++) {
        new_size += tensors[i]->shape.dims[0];
    }
    
    size_t new_dims[] = {new_size, tensors[0]->shape.dims[1]};
    cnp_shape_t new_shape = cnp_shape_create(2, new_dims);
    cnp_tensor_t *result = cnp_tensor_alloc(&new_shape, tensors[0]->dtype);
    
    // Copy data
    if (tensors[0]->dtype == CNP_FLOAT32) {
        float *result_data = (float*)result->data;
        size_t offset = 0;
        
        for (size_t i = 0; i < num_tensors; i++) {
            float *tensor_data = (float*)tensors[i]->data;
            memcpy(&result_data[offset], tensor_data, 
                   cnp_dtype_size(tensors[i]->dtype) * tensors[i]->shape.size);
            offset += tensors[i]->shape.size;
        }
    }
    
    cnp_shape_free(&new_shape);
    return result;
} 