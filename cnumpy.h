#ifndef CNUMPY_H
#define CNUMPY_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct cnp_tensor cnp_tensor_t;
typedef struct cnp_var cnp_var_t;
typedef struct cnp_op cnp_op_t;
typedef struct cnp_optimizer cnp_optimizer_t;

// Execution modes
typedef enum {
    CNP_EAGER_EXECUTION = 1,
    CNP_STATIC_EXECUTION = 2
} cnp_execution_mode_t;

// Data types
typedef enum {
    CNP_FLOAT32,
    CNP_FLOAT64,
    CNP_INT32,
    CNP_INT64
} cnp_dtype_t;

// Tensor shape structure
typedef struct {
    size_t ndim;
    size_t *dims;
    size_t size;  // total number of elements
} cnp_shape_t;

// Base tensor structure
struct cnp_tensor {
    char *name;
    cnp_dtype_t dtype;
    cnp_shape_t shape;
    void *data;
    
    // For gradient computation
    cnp_tensor_t *grad;
    bool requires_grad;
    
    // Reference counting for memory management
    int ref_count;
    
    // For computational graph
    cnp_op_t *op;  // Operation that created this tensor (NULL for variables)
    cnp_tensor_t **inputs;  // Input tensors (for ops)
    size_t num_inputs;
};

// Variable tensor structure
struct cnp_var {
    cnp_tensor_t *tensor;
    bool trainable;
};

// Operation types
typedef enum {
    CNP_OP_ADD,
    CNP_OP_SUB,
    CNP_OP_MUL,
    CNP_OP_MATMUL,
    CNP_OP_NEG,
    CNP_OP_EXP,
    CNP_OP_RELU,
    CNP_OP_SIGMOID,
    CNP_OP_TANH,
    CNP_OP_TRANSPOSE,
    CNP_OP_RESHAPE,
    CNP_OP_REDUCE_SUM,
    CNP_OP_REDUCE_MAX,
    CNP_OP_REDUCE_MIN,
    CNP_OP_CONCAT,
    CNP_OP_INDEX,
    CNP_OP_LINEAR,
    CNP_OP_STOP_GRAD,
    CNP_OP_POW,
    CNP_OP_THRESHOLD,
    CNP_OP_NAN2NUM
} cnp_op_type_t;

// Forward function pointer type
typedef cnp_tensor_t* (*cnp_forward_fn_t)(cnp_tensor_t **inputs, size_t num_inputs, void *params);

// Backward function pointer type
typedef void (*cnp_backward_fn_t)(cnp_tensor_t *output, cnp_tensor_t **inputs, size_t num_inputs, void *params);

// Operation structure
struct cnp_op {
    cnp_op_type_t type;
    cnp_forward_fn_t forward;
    cnp_backward_fn_t backward;
    void *params;  // Operation-specific parameters
};

// Graph management
typedef struct {
    cnp_execution_mode_t mode;
    size_t tensor_count;
    cnp_tensor_t **tensors;
    size_t capacity;
} cnp_graph_t;

// Name scope management
typedef struct {
    char **prefixes;
    size_t num_prefixes;
    size_t capacity;
} cnp_name_scope_t;

// Optimizer structure
struct cnp_optimizer {
    void (*minimize)(cnp_optimizer_t *self, cnp_tensor_t *loss);
    void (*step)(cnp_optimizer_t *self, cnp_tensor_t **params, cnp_tensor_t **grads, size_t num_params);
    void *params;  // Optimizer-specific parameters
};

// SGD optimizer parameters
typedef struct {
    float lr;
    bool debug;
} cnp_sgd_params_t;

// ============================================================================
// Core API Functions
// ============================================================================

// Initialization and cleanup
void cnp_init(void);
void cnp_cleanup(void);

// Graph management
cnp_graph_t* cnp_get_default_graph(void);
void cnp_set_execution_mode(cnp_execution_mode_t mode);
cnp_execution_mode_t cnp_get_execution_mode(void);

// Memory management
cnp_tensor_t* cnp_tensor_alloc(const cnp_shape_t *shape, cnp_dtype_t dtype);
void cnp_tensor_free(cnp_tensor_t *tensor);
void cnp_tensor_incref(cnp_tensor_t *tensor);
void cnp_tensor_decref(cnp_tensor_t *tensor);

// Shape utilities
cnp_shape_t cnp_shape_create(size_t ndim, const size_t *dims);
void cnp_shape_free(cnp_shape_t *shape);
cnp_shape_t cnp_shape_copy(const cnp_shape_t *shape);
bool cnp_shape_equal(const cnp_shape_t *a, const cnp_shape_t *b);
size_t cnp_shape_size(const cnp_shape_t *shape);

// Tensor creation
cnp_tensor_t* cnp_tensor_create(const cnp_shape_t *shape, cnp_dtype_t dtype, const void *data);
cnp_tensor_t* cnp_zeros(const cnp_shape_t *shape, cnp_dtype_t dtype);
cnp_tensor_t* cnp_ones(const cnp_shape_t *shape, cnp_dtype_t dtype);
cnp_tensor_t* cnp_randn(const cnp_shape_t *shape, cnp_dtype_t dtype);
cnp_tensor_t* cnp_uniform(const cnp_shape_t *shape, cnp_dtype_t dtype, float low, float high);

// Variable creation
cnp_var_t* cnp_var_create(const cnp_shape_t *shape, cnp_dtype_t dtype, const void *data, bool trainable);
cnp_var_t* cnp_var_zeros(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable);
cnp_var_t* cnp_var_ones(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable);
cnp_var_t* cnp_var_randn(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable);

// Basic operations
cnp_tensor_t* cnp_add(cnp_tensor_t *a, cnp_tensor_t *b);
cnp_tensor_t* cnp_sub(cnp_tensor_t *a, cnp_tensor_t *b);
cnp_tensor_t* cnp_mul(cnp_tensor_t *a, cnp_tensor_t *b);
cnp_tensor_t* cnp_matmul(cnp_tensor_t *a, cnp_tensor_t *b);
cnp_tensor_t* cnp_neg(cnp_tensor_t *a);
cnp_tensor_t* cnp_pow(cnp_tensor_t *a, float power);

// Activation functions
cnp_tensor_t* cnp_relu(cnp_tensor_t *a);
cnp_tensor_t* cnp_sigmoid(cnp_tensor_t *a);
cnp_tensor_t* cnp_tanh(cnp_tensor_t *a);
cnp_tensor_t* cnp_exp(cnp_tensor_t *a);
cnp_tensor_t* cnp_linear(cnp_tensor_t *a);

// Reduction operations
cnp_tensor_t* cnp_reduce_sum(cnp_tensor_t *a, int axis);
cnp_tensor_t* cnp_reduce_max(cnp_tensor_t *a, int axis);
cnp_tensor_t* cnp_reduce_min(cnp_tensor_t *a, int axis);

// Shape operations
cnp_tensor_t* cnp_transpose(cnp_tensor_t *a, const int *axes);
cnp_tensor_t* cnp_reshape(cnp_tensor_t *a, const cnp_shape_t *new_shape);
cnp_tensor_t* cnp_concat(cnp_tensor_t **tensors, size_t num_tensors, int axis);

// Gradient computation
void cnp_backward(cnp_tensor_t *loss);
void cnp_zero_grad(cnp_tensor_t *tensor);

// Optimizers
cnp_optimizer_t* cnp_sgd_create(float lr, bool debug);
void cnp_optimizer_free(cnp_optimizer_t *optimizer);

// Name scoping
cnp_name_scope_t* cnp_name_scope_create(void);
void cnp_name_scope_free(cnp_name_scope_t *scope);
void cnp_name_scope_push(cnp_name_scope_t *scope, const char *prefix);
void cnp_name_scope_pop(cnp_name_scope_t *scope);
char* cnp_name_scope_get_name(cnp_name_scope_t *scope, const char *base_name);

// Utility functions
void cnp_print_tensor(const cnp_tensor_t *tensor);
void cnp_print_shape(const cnp_shape_t *shape);
size_t cnp_dtype_size(cnp_dtype_t dtype);

// Data access
void* cnp_tensor_data(cnp_tensor_t *tensor);
float cnp_tensor_get_float(cnp_tensor_t *tensor, const size_t *indices);
void cnp_tensor_set_float(cnp_tensor_t *tensor, const size_t *indices, float value);

// Additional utility functions
void cnp_print_computation_graph(cnp_tensor_t *tensor, int depth);
size_t cnp_count_parameters(cnp_tensor_t *tensor);
void cnp_collect_parameters(cnp_tensor_t *tensor, cnp_tensor_t **params, size_t *count, size_t max_count);
void cnp_zero_grad_recursive(cnp_tensor_t *tensor);

// Variable creation using uniform distribution
cnp_var_t* cnp_var_uniform(const cnp_shape_t *shape, cnp_dtype_t dtype, float low, float high, bool trainable);

// Variable cleanup
void cnp_var_free(cnp_var_t *var);

#ifdef __cplusplus
}
#endif

#endif // CNUMPY_H 