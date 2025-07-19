#include "cnumpy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Forward declarations
static void cnp_sgd_update_parameters(cnp_tensor_t *tensor, cnp_sgd_params_t *params);
static void cnp_backward_recursive(cnp_tensor_t *tensor);

// ============================================================================
// Name scope implementation
// ============================================================================

cnp_name_scope_t* cnp_name_scope_create(void) {
    cnp_name_scope_t *scope = malloc(sizeof(cnp_name_scope_t));
    scope->prefixes = NULL;
    scope->num_prefixes = 0;
    scope->capacity = 0;
    return scope;
}

void cnp_name_scope_free(cnp_name_scope_t *scope) {
    if (!scope) return;
    
    // Free all prefix strings
    for (size_t i = 0; i < scope->num_prefixes; i++) {
        free(scope->prefixes[i]);
    }
    
    // Free prefix array
    if (scope->prefixes) {
        free(scope->prefixes);
    }
    
    free(scope);
}

void cnp_name_scope_push(cnp_name_scope_t *scope, const char *prefix) {
    if (!scope || !prefix) return;
    
    // Expand capacity if needed
    if (scope->num_prefixes >= scope->capacity) {
        scope->capacity = scope->capacity == 0 ? 4 : scope->capacity * 2;
        scope->prefixes = realloc(scope->prefixes, sizeof(char*) * scope->capacity);
    }
    
    // Add new prefix
    scope->prefixes[scope->num_prefixes] = malloc(strlen(prefix) + 1);
    strcpy(scope->prefixes[scope->num_prefixes], prefix);
    scope->num_prefixes++;
}

void cnp_name_scope_pop(cnp_name_scope_t *scope) {
    if (!scope || scope->num_prefixes == 0) return;
    
    // Free the last prefix
    free(scope->prefixes[scope->num_prefixes - 1]);
    scope->num_prefixes--;
}

char* cnp_name_scope_get_name(cnp_name_scope_t *scope, const char *base_name) {
    if (!scope || !base_name) {
        // Just return a copy of base_name
        char *result = malloc(strlen(base_name) + 1);
        strcpy(result, base_name);
        return result;
    }
    
    // Calculate total length needed
    size_t total_length = strlen(base_name) + 1;  // +1 for null terminator
    
    for (size_t i = 0; i < scope->num_prefixes; i++) {
        total_length += strlen(scope->prefixes[i]) + 1;  // +1 for colon
    }
    
    // Allocate and build the scoped name
    char *result = malloc(total_length);
    result[0] = '\0';
    
    // Add prefixes
    for (size_t i = 0; i < scope->num_prefixes; i++) {
        strcat(result, scope->prefixes[i]);
        strcat(result, ":");
    }
    
    // Add base name
    strcat(result, base_name);
    
    return result;
}

// ============================================================================
// SGD Optimizer implementation
// ============================================================================

static void cnp_sgd_minimize(cnp_optimizer_t *self, cnp_tensor_t *loss) {
    cnp_sgd_params_t *params = (cnp_sgd_params_t*)self->params;
    
    if (params->debug) {
        printf("SGD: Minimizing loss\n");
        cnp_print_tensor(loss);
    }
    
    // Initialize loss gradient to ones
    if (!loss->grad) {
        loss->grad = cnp_ones(&loss->shape, loss->dtype);
        cnp_tensor_incref(loss->grad);  // Increment reference count
    }
    
    // First, compute all gradients via backward pass
    cnp_backward(loss);
    
    // Then, update parameters
    cnp_sgd_update_parameters(loss, params);
}

static void cnp_sgd_update_parameters(cnp_tensor_t *tensor, cnp_sgd_params_t *params) {
    if (!tensor) return;
    
    // If this is a variable tensor with gradients, update its parameters
    if (!tensor->op && tensor->requires_grad && tensor->grad) {
        if (params->debug) {
            printf("SGD: Updating tensor %s\n", tensor->name ? tensor->name : "unnamed");
            printf("SGD: Before update: ");
            cnp_print_tensor(tensor);
            printf("SGD: Gradient: ");
            cnp_print_tensor(tensor->grad);
        }
        
        // Update parameters: param = param - lr * grad
        if (tensor->dtype == CNP_FLOAT32) {
            float *data = (float*)tensor->data;
            float *grad = (float*)tensor->grad->data;
            
            for (size_t i = 0; i < tensor->shape.size; i++) {
                data[i] -= params->lr * grad[i];
            }
        } else if (tensor->dtype == CNP_FLOAT64) {
            double *data = (double*)tensor->data;
            double *grad = (double*)tensor->grad->data;
            
            for (size_t i = 0; i < tensor->shape.size; i++) {
                data[i] -= params->lr * grad[i];
            }
        }
        
        if (params->debug) {
            printf("SGD: After update: ");
            cnp_print_tensor(tensor);
        }
    }
    
    // Recursively update parameters in input tensors
    for (size_t i = 0; i < tensor->num_inputs; i++) {
        cnp_sgd_update_parameters(tensor->inputs[i], params);
    }
}

static void cnp_sgd_step(cnp_optimizer_t *self, cnp_tensor_t **params_tensors, cnp_tensor_t **grads, size_t num_params) {
    cnp_sgd_params_t *params = (cnp_sgd_params_t*)self->params;
    
    for (size_t i = 0; i < num_params; i++) {
        cnp_tensor_t *param = params_tensors[i];
        cnp_tensor_t *grad = grads[i];
        
        if (param->dtype == CNP_FLOAT32) {
            float *param_data = (float*)param->data;
            float *grad_data = (float*)grad->data;
            
            for (size_t j = 0; j < param->shape.size; j++) {
                param_data[j] -= params->lr * grad_data[j];
            }
        } else if (param->dtype == CNP_FLOAT64) {
            double *param_data = (double*)param->data;
            double *grad_data = (double*)grad->data;
            
            for (size_t j = 0; j < param->shape.size; j++) {
                param_data[j] -= params->lr * grad_data[j];
            }
        }
    }
}

cnp_optimizer_t* cnp_sgd_create(float lr, bool debug) {
    cnp_optimizer_t *optimizer = malloc(sizeof(cnp_optimizer_t));
    
    // Set function pointers
    optimizer->minimize = cnp_sgd_minimize;
    optimizer->step = cnp_sgd_step;
    
    // Set parameters
    cnp_sgd_params_t *params = malloc(sizeof(cnp_sgd_params_t));
    params->lr = lr;
    params->debug = debug;
    optimizer->params = params;
    
    return optimizer;
}

void cnp_optimizer_free(cnp_optimizer_t *optimizer) {
    if (!optimizer) return;
    
    if (optimizer->params) {
        free(optimizer->params);
    }
    
    free(optimizer);
}

// ============================================================================
// Enhanced automatic differentiation
// ============================================================================

void cnp_backward(cnp_tensor_t *loss) {
    if (!loss) return;
    
    // Initialize gradient if not present
    if (!loss->grad) {
        loss->grad = cnp_ones(&loss->shape, loss->dtype);
    }
    
    // Perform topological sort and backward pass
    cnp_backward_recursive(loss);
}

static void cnp_backward_recursive(cnp_tensor_t *tensor) {
    if (!tensor) return;
    
    // If this is a leaf tensor (variable), no further backpropagation needed
    if (!tensor->op) return;
    
    // Call the operation's backward function to compute gradients for inputs
    if (tensor->op->backward && tensor->grad) {
        tensor->op->backward(tensor, tensor->inputs, tensor->num_inputs, tensor->op->params);
    }
    
    // Recursively backpropagate to input tensors
    for (size_t i = 0; i < tensor->num_inputs; i++) {
        cnp_backward_recursive(tensor->inputs[i]);
    }
}

void cnp_zero_grad_recursive(cnp_tensor_t *tensor) {
    if (!tensor) return;
    
    // Zero out this tensor's gradient
    cnp_zero_grad(tensor);
    
    // Recursively zero gradients of input tensors
    for (size_t i = 0; i < tensor->num_inputs; i++) {
        cnp_zero_grad_recursive(tensor->inputs[i]);
    }
}

// ============================================================================
// Utility functions for graph traversal
// ============================================================================

void cnp_print_computation_graph(cnp_tensor_t *tensor, int depth) {
    if (!tensor) return;
    
    // Print indentation
    for (int i = 0; i < depth; i++) {
        printf("  ");
    }
    
    // Print tensor info
    printf("Tensor: %s, shape: ", tensor->name ? tensor->name : "unnamed");
    cnp_print_shape(&tensor->shape);
    
    if (tensor->op) {
        printf(", op: %d", tensor->op->type);
    }
    
    if (tensor->requires_grad) {
        printf(", requires_grad: true");
    }
    
    printf("\n");
    
    // Print input tensors
    for (size_t i = 0; i < tensor->num_inputs; i++) {
        cnp_print_computation_graph(tensor->inputs[i], depth + 1);
    }
}

size_t cnp_count_parameters(cnp_tensor_t *tensor) {
    if (!tensor) return 0;
    
    size_t count = 0;
    
    // If this is a trainable parameter, count it
    if (tensor->requires_grad && !tensor->op) {
        count += tensor->shape.size;
    }
    
    // Recursively count parameters in input tensors
    for (size_t i = 0; i < tensor->num_inputs; i++) {
        count += cnp_count_parameters(tensor->inputs[i]);
    }
    
    return count;
}

void cnp_collect_parameters(cnp_tensor_t *tensor, cnp_tensor_t **params, size_t *count, size_t max_count) {
    if (!tensor || *count >= max_count) return;
    
    // If this is a trainable parameter, add it to the list
    if (tensor->requires_grad && !tensor->op) {
        params[*count] = tensor;
        (*count)++;
    }
    
    // Recursively collect parameters from input tensors
    for (size_t i = 0; i < tensor->num_inputs && *count < max_count; i++) {
        cnp_collect_parameters(tensor->inputs[i], params, count, max_count);
    }
} 