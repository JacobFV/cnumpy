#include "../cnumpy.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("CNmpy Debug Test\n");
    printf("================\n\n");
    
    // Initialize the library
    cnp_init();
    
    // Create a simple variable with non-zero initial value
    size_t dims[] = {1};
    cnp_shape_t shape = cnp_shape_create(1, dims);
    
    float initial_value = 2.0f;
    cnp_var_t *x = cnp_var_create(&shape, CNP_FLOAT32, &initial_value, true);
    
    printf("Initial x: ");
    cnp_print_tensor(x->tensor);
    printf("x requires_grad: %s\n", x->tensor->requires_grad ? "true" : "false");
    
    // Create a simple loss: loss = x^2
    cnp_tensor_t *loss = cnp_mul(x->tensor, x->tensor);
    
    printf("Loss: ");
    cnp_print_tensor(loss);
    printf("Loss requires_grad: %s\n", loss->requires_grad ? "true" : "false");
    
    // Check the computational graph
    printf("\nComputational graph:\n");
    cnp_print_computation_graph(loss, 0);
    
    // Initialize loss gradient
    if (!loss->grad) {
        loss->grad = cnp_ones(&loss->shape, loss->dtype);
        cnp_tensor_incref(loss->grad);  // Increment reference count
    }
    
    printf("\nLoss gradient: ");
    cnp_print_tensor(loss->grad);
    
    // Call backward pass
    printf("\nCalling backward pass...\n");
    cnp_backward(loss);
    
    // Check if x has gradients
    printf("After backward pass:\n");
    if (x->tensor->grad) {
        printf("x gradient: ");
        cnp_print_tensor(x->tensor->grad);
    } else {
        printf("x gradient: NULL\n");
    }
    
    // Try to update x manually
    if (x->tensor->grad) {
        printf("\nManual update of x:\n");
        float *x_data = (float*)x->tensor->data;
        float *x_grad = (float*)x->tensor->grad->data;
        
        printf("Before: x = %f, grad = %f\n", x_data[0], x_grad[0]);
        x_data[0] -= 0.1f * x_grad[0];  // lr = 0.1
        printf("After: x = %f\n", x_data[0]);
    }
    
    // Cleanup
    cnp_shape_free(&shape);
    cnp_cleanup();
    
    return 0;
} 