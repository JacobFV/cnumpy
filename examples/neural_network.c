#include "../src/cnumpy.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple neural network structure
typedef struct {
    cnp_var_t *W1;  // First layer weights
    cnp_var_t *b1;  // First layer bias
    cnp_var_t *W2;  // Second layer weights
    cnp_var_t *b2;  // Second layer bias
} simple_nn_t;

// Create a simple 2-layer neural network
simple_nn_t* create_simple_nn(size_t input_size, size_t hidden_size, size_t output_size) {
    simple_nn_t *nn = malloc(sizeof(simple_nn_t));
    
    // Create weight and bias tensors
    size_t w1_dims[] = {input_size, hidden_size};
    size_t b1_dims[] = {1, hidden_size};
    size_t w2_dims[] = {hidden_size, output_size};
    size_t b2_dims[] = {1, output_size};
    
    cnp_shape_t w1_shape = cnp_shape_create(2, w1_dims);
    cnp_shape_t b1_shape = cnp_shape_create(2, b1_dims);
    cnp_shape_t w2_shape = cnp_shape_create(2, w2_dims);
    cnp_shape_t b2_shape = cnp_shape_create(2, b2_dims);
    
    // Initialize with small random values
    nn->W1 = cnp_var_uniform(&w1_shape, CNP_FLOAT32, -0.1f, 0.1f, true);
    nn->b1 = cnp_var_zeros(&b1_shape, CNP_FLOAT32, true);
    nn->W2 = cnp_var_uniform(&w2_shape, CNP_FLOAT32, -0.1f, 0.1f, true);
    nn->b2 = cnp_var_zeros(&b2_shape, CNP_FLOAT32, true);
    
    cnp_shape_free(&w1_shape);
    cnp_shape_free(&b1_shape);
    cnp_shape_free(&w2_shape);
    cnp_shape_free(&b2_shape);
    
    return nn;
}

// Forward pass through the network
cnp_tensor_t* forward(simple_nn_t *nn, cnp_tensor_t *input) {
    // Layer 1: input @ W1 + b1
    cnp_tensor_t *z1 = cnp_matmul(input, nn->W1->tensor);
    cnp_tensor_t *a1 = cnp_add(z1, nn->b1->tensor);
    
    // Apply ReLU activation
    cnp_tensor_t *h1 = cnp_relu(a1);
    
    // Layer 2: h1 @ W2 + b2
    cnp_tensor_t *z2 = cnp_matmul(h1, nn->W2->tensor);
    cnp_tensor_t *output = cnp_add(z2, nn->b2->tensor);
    
    return output;
}

// Compute mean squared error loss
cnp_tensor_t* mse_loss(cnp_tensor_t *predictions, cnp_tensor_t *targets) {
    cnp_tensor_t *diff = cnp_sub(predictions, targets);
    cnp_tensor_t *squared = cnp_mul(diff, diff);
    cnp_tensor_t *loss = cnp_reduce_sum(squared, -1);
    return loss;
}

// Free neural network resources
void free_simple_nn(simple_nn_t *nn) {
    if (nn) {
        // Note: In a real implementation, you'd want to properly manage
        // the memory of the tensor variables
        free(nn);
    }
}

int main() {
    printf("Neural Network Example\n");
    printf("=====================\n\n");
    
    // Initialize the library
    cnp_init();
    
    // Create a simple neural network
    // Architecture: 2 -> 4 -> 1 (input -> hidden -> output)
    simple_nn_t *nn = create_simple_nn(2, 4, 1);
    
    printf("Created neural network with architecture 2 -> 4 -> 1\n");
    printf("W1 shape: ");
    cnp_print_shape(&nn->W1->tensor->shape);
    printf("\nW2 shape: ");
    cnp_print_shape(&nn->W2->tensor->shape);
    printf("\n\n");
    
    // Create some training data
    // We'll try to learn the XOR function
    float input_data[] = {
        0.0f, 0.0f,  // Input 1
        0.0f, 1.0f,  // Input 2
        1.0f, 0.0f,  // Input 3
        1.0f, 1.0f   // Input 4
    };
    
    float target_data[] = {
        0.0f,  // XOR(0,0) = 0
        1.0f,  // XOR(0,1) = 1
        1.0f,  // XOR(1,0) = 1
        0.0f   // XOR(1,1) = 0
    };
    
    size_t input_dims[] = {4, 2};
    size_t target_dims[] = {4, 1};
    cnp_shape_t input_shape = cnp_shape_create(2, input_dims);
    cnp_shape_t target_shape = cnp_shape_create(2, target_dims);
    
    cnp_tensor_t *inputs = cnp_tensor_create(&input_shape, CNP_FLOAT32, input_data);
    cnp_tensor_t *targets = cnp_tensor_create(&target_shape, CNP_FLOAT32, target_data);
    
    printf("Training data:\n");
    printf("Inputs: ");
    cnp_print_tensor(inputs);
    printf("Targets: ");
    cnp_print_tensor(targets);
    printf("\n");
    
    // Create optimizer
    cnp_optimizer_t *optimizer = cnp_sgd_create(0.01f, false);
    
    // Training loop
    int epochs = 100;
    printf("Training for %d epochs...\n", epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        cnp_tensor_t *predictions = forward(nn, inputs);
        
        // Compute loss
        cnp_tensor_t *loss = mse_loss(predictions, targets);
        
        // Print progress every 20 epochs
        if (epoch % 20 == 0) {
            printf("Epoch %d - Loss: ", epoch);
            cnp_print_tensor(loss);
            printf("Predictions: ");
            cnp_print_tensor(predictions);
        }
        
        // Backward pass
        optimizer->minimize(optimizer, loss);
        
        // Zero gradients for next iteration
        cnp_zero_grad(nn->W1->tensor);
        cnp_zero_grad(nn->b1->tensor);
        cnp_zero_grad(nn->W2->tensor);
        cnp_zero_grad(nn->b2->tensor);
    }
    
    // Test the trained network
    printf("\nTesting trained network:\n");
    cnp_tensor_t *final_predictions = forward(nn, inputs);
    
    printf("Final predictions: ");
    cnp_print_tensor(final_predictions);
    printf("True targets: ");
    cnp_print_tensor(targets);
    
    // Test individual examples
    printf("\nIndividual test cases:\n");
    
    float test_cases[][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    float expected[] = {0.0f, 1.0f, 1.0f, 0.0f};
    
    for (int i = 0; i < 4; i++) {
        size_t single_dims[] = {1, 2};
        cnp_shape_t single_shape = cnp_shape_create(2, single_dims);
        
        cnp_tensor_t *single_input = cnp_tensor_create(&single_shape, CNP_FLOAT32, test_cases[i]);
        cnp_tensor_t *single_prediction = forward(nn, single_input);
        
        float pred_value = cnp_tensor_get_float(single_prediction, (size_t[]){0, 0});
        
        printf("XOR(%.0f, %.0f) = %.3f (expected %.0f)\n", 
               test_cases[i][0], test_cases[i][1], pred_value, expected[i]);
        
        cnp_shape_free(&single_shape);
    }
    
    // Demonstrate computation graph
    printf("\nComputation graph for one forward pass:\n");
    cnp_tensor_t *graph_example = forward(nn, inputs);
    cnp_print_computation_graph(graph_example, 0);
    
    // Parameter count
    printf("\nNetwork parameters:\n");
    size_t param_count = cnp_count_parameters(nn->W1->tensor) + 
                        cnp_count_parameters(nn->b1->tensor) + 
                        cnp_count_parameters(nn->W2->tensor) + 
                        cnp_count_parameters(nn->b2->tensor);
    
    printf("Total parameters: %zu\n", param_count);
    
    // Show learned weights
    printf("\nLearned weights:\n");
    printf("W1: ");
    cnp_print_tensor(nn->W1->tensor);
    printf("b1: ");
    cnp_print_tensor(nn->b1->tensor);
    printf("W2: ");
    cnp_print_tensor(nn->W2->tensor);
    printf("b2: ");
    cnp_print_tensor(nn->b2->tensor);
    
    // Cleanup
    cnp_shape_free(&input_shape);
    cnp_shape_free(&target_shape);
    cnp_optimizer_free(optimizer);
    free_simple_nn(nn);
    
    cnp_cleanup();
    
    printf("\nNeural network example completed successfully!\n");
    
    return 0;
} 