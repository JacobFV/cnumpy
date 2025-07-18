#include "../cnumpy.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int main() {
    printf("CNmpy Basic Test\n");
    printf("================\n\n");
    
    // Initialize the library
    cnp_init();
    
    // Test 1: Basic tensor creation
    printf("Test 1: Basic tensor creation\n");
    size_t dims[] = {2, 3};
    cnp_shape_t shape = cnp_shape_create(2, dims);
    
    cnp_tensor_t *a = cnp_ones(&shape, CNP_FLOAT32);
    cnp_tensor_t *b = cnp_ones(&shape, CNP_FLOAT32);
    
    printf("Tensor a: ");
    cnp_print_tensor(a);
    printf("Tensor b: ");
    cnp_print_tensor(b);
    
    // Test 2: Basic arithmetic operations
    printf("\nTest 2: Basic arithmetic operations\n");
    cnp_tensor_t *c = cnp_add(a, b);
    printf("a + b = ");
    cnp_print_tensor(c);
    
    cnp_tensor_t *d = cnp_mul(a, b);
    printf("a * b = ");
    cnp_print_tensor(d);
    
    // Test 3: Activation functions
    printf("\nTest 3: Activation functions\n");
    
    // Create a tensor with some values
    float data[] = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 0.5f};
    size_t test_dims[] = {2, 3};
    cnp_shape_t test_shape = cnp_shape_create(2, test_dims);
    cnp_tensor_t *test_tensor = cnp_tensor_create(&test_shape, CNP_FLOAT32, data);
    
    printf("Input tensor: ");
    cnp_print_tensor(test_tensor);
    
    cnp_tensor_t *relu_result = cnp_relu(test_tensor);
    printf("ReLU result: ");
    cnp_print_tensor(relu_result);
    
    cnp_tensor_t *sigmoid_result = cnp_sigmoid(test_tensor);
    printf("Sigmoid result: ");
    cnp_print_tensor(sigmoid_result);
    
    cnp_tensor_t *tanh_result = cnp_tanh(test_tensor);
    printf("Tanh result: ");
    cnp_print_tensor(tanh_result);
    
    // Test 4: Matrix multiplication
    printf("\nTest 4: Matrix multiplication\n");
    size_t mat_dims1[] = {2, 3};
    size_t mat_dims2[] = {3, 2};
    cnp_shape_t mat_shape1 = cnp_shape_create(2, mat_dims1);
    cnp_shape_t mat_shape2 = cnp_shape_create(2, mat_dims2);
    
    cnp_tensor_t *mat1 = cnp_ones(&mat_shape1, CNP_FLOAT32);
    cnp_tensor_t *mat2 = cnp_ones(&mat_shape2, CNP_FLOAT32);
    
    printf("Matrix 1 (2x3): ");
    cnp_print_tensor(mat1);
    printf("Matrix 2 (3x2): ");
    cnp_print_tensor(mat2);
    
    cnp_tensor_t *mat_result = cnp_matmul(mat1, mat2);
    printf("Matrix multiplication result (2x2): ");
    cnp_print_tensor(mat_result);
    
    // Test 5: Variable creation and gradient tracking
    printf("\nTest 5: Variable creation and gradient tracking\n");
    size_t var_dims[] = {2, 2};
    cnp_shape_t var_shape = cnp_shape_create(2, var_dims);
    
    cnp_var_t *var1 = cnp_var_ones(&var_shape, CNP_FLOAT32, true);  // trainable
    cnp_var_t *var2 = cnp_var_ones(&var_shape, CNP_FLOAT32, false); // not trainable
    
    printf("Trainable variable: ");
    cnp_print_tensor(var1->tensor);
    printf("Requires grad: %s\n", var1->tensor->requires_grad ? "true" : "false");
    
    printf("Non-trainable variable: ");
    cnp_print_tensor(var2->tensor);
    printf("Requires grad: %s\n", var2->tensor->requires_grad ? "true" : "false");
    
    // Test 6: Simple optimization
    printf("\nTest 6: Simple optimization\n");
    
    // Create a simple loss function: loss = (x - target)^2
    float target_data[] = {2.0f, 3.0f};
    size_t target_dims[] = {2};
    cnp_shape_t target_shape = cnp_shape_create(1, target_dims);
    
    cnp_tensor_t *target = cnp_tensor_create(&target_shape, CNP_FLOAT32, target_data);
    cnp_var_t *x = cnp_var_zeros(&target_shape, CNP_FLOAT32, true);
    
    printf("Initial x: ");
    cnp_print_tensor(x->tensor);
    printf("Target: ");
    cnp_print_tensor(target);
    
    // Create optimizer
    cnp_optimizer_t *optimizer = cnp_sgd_create(0.1f, true);
    
    // Simple training loop
    for (int epoch = 0; epoch < 5; epoch++) {
        // Forward pass: loss = (x - target)^2
        cnp_tensor_t *diff = cnp_sub(x->tensor, target);
        cnp_tensor_t *loss = cnp_mul(diff, diff);
        cnp_tensor_t *loss_sum = cnp_reduce_sum(loss, -1);
        
        printf("Epoch %d - Loss: ", epoch);
        cnp_print_tensor(loss_sum);
        
        // Backward pass
        optimizer->minimize(optimizer, loss_sum);
        
        printf("Updated x: ");
        cnp_print_tensor(x->tensor);
        
        // Clean up gradients for next iteration
        cnp_zero_grad(x->tensor);
    }
    
    // Test 7: Memory management
    printf("\nTest 7: Memory management\n");
    printf("Reference counting test...\n");
    
    cnp_tensor_t *ref_test = cnp_ones(&shape, CNP_FLOAT32);
    printf("Initial ref count: %d\n", ref_test->ref_count);
    
    cnp_tensor_incref(ref_test);
    printf("After incref: %d\n", ref_test->ref_count);
    
    cnp_tensor_decref(ref_test);
    printf("After decref: %d\n", ref_test->ref_count);
    
    // Test 8: Shape operations
    printf("\nTest 8: Shape operations\n");
    
    size_t reshape_dims[] = {3, 2};
    cnp_shape_t reshape_shape = cnp_shape_create(2, reshape_dims);
    
    cnp_tensor_t *reshaped = cnp_reshape(test_tensor, &reshape_shape);
    printf("Original shape: ");
    cnp_print_shape(&test_tensor->shape);
    printf("\nReshaped: ");
    cnp_print_tensor(reshaped);
    
    // Test 9: Computation graph
    printf("\nTest 9: Computation graph\n");
    cnp_tensor_t *graph_test = cnp_add(cnp_mul(a, b), cnp_relu(test_tensor));
    printf("Computation graph:\n");
    cnp_print_computation_graph(graph_test, 0);
    
    printf("\nAll tests completed successfully!\n");
    
    // Cleanup
    cnp_shape_free(&shape);
    cnp_shape_free(&test_shape);
    cnp_shape_free(&mat_shape1);
    cnp_shape_free(&mat_shape2);
    cnp_shape_free(&var_shape);
    cnp_shape_free(&target_shape);
    cnp_shape_free(&reshape_shape);
    
    cnp_optimizer_free(optimizer);
    
    cnp_cleanup();
    
    return 0;
} 