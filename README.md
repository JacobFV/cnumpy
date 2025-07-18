# CNmpy - C Implementation of JNumpy

A C implementation of the core functionality from Jacob's NumPy library for machine learning, providing tensors, automatic differentiation, and basic neural network operations.

## Features

- **Tensor Operations**: Basic tensor creation, arithmetic operations, and shape manipulation
- **Automatic Differentiation**: Reverse-mode automatic differentiation for gradient computation
- **Neural Network Operations**: Matrix multiplication, activation functions (ReLU, Sigmoid, Tanh)
- **Optimization**: SGD optimizer with customizable learning rates
- **Memory Management**: Reference counting for automatic memory management
- **Computation Graphs**: Build and traverse computational graphs for complex operations

## Architecture

The library consists of several main components:

- **Core (`cnumpy_core.c`)**: Tensor creation, memory management, and basic utilities
- **Operations (`cnumpy_ops.c`)**: Mathematical operations and activation functions
- **Scope (`cnumpy_scope.c`)**: Name scoping, automatic differentiation, and optimizers
- **Header (`cnumpy.h`)**: Complete API definitions and data structures

## Data Structures

### Tensor (`cnp_tensor_t`)
- Multi-dimensional arrays with shape information
- Support for float32, float64, int32, and int64 data types
- Automatic gradient tracking for differentiable operations
- Reference counting for memory management

### Variables (`cnp_var_t`)
- Trainable parameters that can be optimized
- Built on top of tensors with additional training state

### Operations (`cnp_op_t`)
- Computational nodes in the graph
- Forward and backward function pointers
- Parameter storage for operation-specific data

## Quick Start

### Building the Library

```bash
# Clone and build
cd cnumpy
make

# Run tests
make test

# Install system-wide (optional)
sudo make install
```

### Basic Usage

```c
#include "cnumpy.h"

int main() {
    // Initialize library
    cnp_init();
    
    // Create tensors
    size_t dims[] = {2, 3};
    cnp_shape_t shape = cnp_shape_create(2, dims);
    
    cnp_tensor_t *a = cnp_ones(&shape, CNP_FLOAT32);
    cnp_tensor_t *b = cnp_ones(&shape, CNP_FLOAT32);
    
    // Basic operations
    cnp_tensor_t *c = cnp_add(a, b);
    cnp_tensor_t *d = cnp_mul(a, b);
    
    // Activation functions
    cnp_tensor_t *relu_result = cnp_relu(a);
    cnp_tensor_t *sigmoid_result = cnp_sigmoid(a);
    
    // Print results
    cnp_print_tensor(c);
    
    // Cleanup
    cnp_shape_free(&shape);
    cnp_cleanup();
    
    return 0;
}
```

### Neural Network Example

```c
#include "cnumpy.h"

// Create trainable variables
cnp_var_t *W = cnp_var_randn(&weight_shape, CNP_FLOAT32, true);
cnp_var_t *b = cnp_var_zeros(&bias_shape, CNP_FLOAT32, true);

// Forward pass
cnp_tensor_t *z = cnp_matmul(input, &W->tensor);
cnp_tensor_t *a = cnp_add(z, &b->tensor);
cnp_tensor_t *output = cnp_relu(a);

// Loss computation
cnp_tensor_t *loss = cnp_mse_loss(output, targets);

// Create and use optimizer
cnp_optimizer_t *optimizer = cnp_sgd_create(0.01f, false);
optimizer->minimize(optimizer, loss);
```

## API Reference

### Tensor Creation

```c
// Create tensors with specific values
cnp_tensor_t* cnp_zeros(const cnp_shape_t *shape, cnp_dtype_t dtype);
cnp_tensor_t* cnp_ones(const cnp_shape_t *shape, cnp_dtype_t dtype);
cnp_tensor_t* cnp_randn(const cnp_shape_t *shape, cnp_dtype_t dtype);
cnp_tensor_t* cnp_uniform(const cnp_shape_t *shape, cnp_dtype_t dtype, float low, float high);

// Create variables (trainable parameters)
cnp_var_t* cnp_var_zeros(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable);
cnp_var_t* cnp_var_ones(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable);
cnp_var_t* cnp_var_randn(const cnp_shape_t *shape, cnp_dtype_t dtype, bool trainable);
```

### Mathematical Operations

```c
// Basic arithmetic
cnp_tensor_t* cnp_add(cnp_tensor_t *a, cnp_tensor_t *b);
cnp_tensor_t* cnp_sub(cnp_tensor_t *a, cnp_tensor_t *b);
cnp_tensor_t* cnp_mul(cnp_tensor_t *a, cnp_tensor_t *b);
cnp_tensor_t* cnp_matmul(cnp_tensor_t *a, cnp_tensor_t *b);
cnp_tensor_t* cnp_pow(cnp_tensor_t *a, float power);

// Activation functions
cnp_tensor_t* cnp_relu(cnp_tensor_t *a);
cnp_tensor_t* cnp_sigmoid(cnp_tensor_t *a);
cnp_tensor_t* cnp_tanh(cnp_tensor_t *a);
cnp_tensor_t* cnp_exp(cnp_tensor_t *a);

// Reduction operations
cnp_tensor_t* cnp_reduce_sum(cnp_tensor_t *a, int axis);
cnp_tensor_t* cnp_reduce_max(cnp_tensor_t *a, int axis);
cnp_tensor_t* cnp_reduce_min(cnp_tensor_t *a, int axis);
```

### Automatic Differentiation

```c
// Compute gradients
void cnp_backward(cnp_tensor_t *loss);

// Zero gradients
void cnp_zero_grad(cnp_tensor_t *tensor);

// Optimization
cnp_optimizer_t* cnp_sgd_create(float lr, bool debug);
void cnp_optimizer_minimize(cnp_optimizer_t *optimizer, cnp_tensor_t *loss);
```

### Shape Operations

```c
// Shape management
cnp_shape_t cnp_shape_create(size_t ndim, const size_t *dims);
void cnp_shape_free(cnp_shape_t *shape);
bool cnp_shape_equal(const cnp_shape_t *a, const cnp_shape_t *b);

// Tensor reshaping
cnp_tensor_t* cnp_reshape(cnp_tensor_t *a, const cnp_shape_t *new_shape);
cnp_tensor_t* cnp_transpose(cnp_tensor_t *a, const int *axes);
```

## Examples

See the `examples/` directory for complete examples:

- **`test_basic.c`**: Basic functionality demonstration
- **`neural_network.c`**: Simple neural network training XOR function

## Building and Testing

```bash
# Build library and examples
make

# Build with debug symbols
make debug

# Build optimized release
make release

# Run tests
make test

# Check for memory leaks (requires valgrind)
make memcheck

# Clean build artifacts
make clean
```

## Data Types

The library supports multiple data types:

- `CNP_FLOAT32`: 32-bit floating point
- `CNP_FLOAT64`: 64-bit floating point  
- `CNP_INT32`: 32-bit signed integer
- `CNP_INT64`: 64-bit signed integer

## Memory Management

CNmpy uses reference counting for automatic memory management:

- Tensors are automatically freed when their reference count reaches zero
- Use `cnp_tensor_incref()` and `cnp_tensor_decref()` for manual reference management
- Call `cnp_cleanup()` at program end to free global resources

## Limitations

Current implementation limitations:

- Broadcasting is not fully implemented
- Only basic reduction operations (sum, max, min) are supported
- Matrix multiplication is limited to 2D tensors
- Gradient computation uses recursive traversal (not optimized for large graphs)
- No GPU support

## Contributing

This is a basic implementation for educational purposes. The codebase is designed to be simple and readable rather than highly optimized.

## License

Released under the MIT License. See LICENSE file for details.

## Relationship to JNumpy

This C implementation mirrors the core functionality of Jacob's NumPy library (jnumpy), providing similar tensor operations and automatic differentiation capabilities in C for performance-critical applications.
