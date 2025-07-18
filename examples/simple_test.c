#include "../cnumpy.h"
#include <stdio.h>

int main() {
    printf("Simple Test\n");
    
    // Initialize
    cnp_init();
    
    // Create a single tensor
    size_t dims[] = {1};
    cnp_shape_t shape = cnp_shape_create(1, dims);
    cnp_tensor_t *a = cnp_zeros(&shape, CNP_FLOAT32);
    
    printf("Created tensor a\n");
    cnp_print_tensor(a);
    
    // Create a variable
    cnp_var_t *var = cnp_var_zeros(&shape, CNP_FLOAT32, true);
    printf("Created variable\n");
    cnp_print_tensor(var->tensor);
    
    // Free the variable
    cnp_var_free(var);
    printf("Freed variable\n");
    
    // Cleanup
    cnp_shape_free(&shape);
    cnp_cleanup();
    printf("Cleaned up\n");
    
    return 0;
} 