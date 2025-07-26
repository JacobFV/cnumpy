#include <stdio.h>
#include <stdlib.h>
#include "../src/cnumpy.h"
#include "../src/rl/cnumpy_rl.h"

int main() {
    printf("Testing CartPole Environment...\n");
    
    // Initialize
    cnp_rl_init();
    
    // Create CartPole environment
    cnp_rl_env_t *env = cnp_rl_cartpole_create();
    if (!env) {
        printf("Failed to create CartPole environment\n");
        return 1;
    }
    
    printf("Created CartPole environment: %s\n", env->name);
    printf("Observation shape: %zu dimensions\n", env->obs_shape.ndim);
    printf("Action shape: %zu dimensions\n", env->action_shape.ndim);
    printf("Number of actions: %zu\n", env->num_actions);
    
    // Test reset
    printf("\nTesting environment reset...\n");
    cnp_rl_step_t *initial_step = env->reset(env);
    if (!initial_step) {
        printf("Failed to reset environment\n");
        cnp_rl_env_decref(env);
        return 1;
    }
    
    printf("Initial observation shape: %zu dimensions\n", initial_step->obs->shape.ndim);
    printf("Initial observation data: [");
    for (size_t i = 0; i < initial_step->obs->shape.size; i++) {
        printf("%.3f", ((float*)initial_step->obs->data)[i]);
        if (i < initial_step->obs->shape.size - 1) printf(", ");
    }
    printf("]\n");
    
    // Test a few steps
    printf("\nTesting environment steps...\n");
    for (int i = 0; i < 5; i++) {
        // Create action (0 or 1)
        float action_data = (float)(i % 2);
        cnp_tensor_t *action = cnp_tensor_create(
            &env->action_shape, CNP_FLOAT32, &action_data
        );
        
        printf("Step %d: Action = %.0f\n", i, action_data);
        
        // Take step
        cnp_rl_step_t *step = env->step(env, action);
        if (!step) {
            printf("Failed to take step\n");
            cnp_tensor_decref(action);
            break;
        }
        
        // Print results
        printf("  Observation: [");
        for (size_t j = 0; j < step->next_obs->shape.size; j++) {
            printf("%.3f", ((float*)step->next_obs->data)[j]);
            if (j < step->next_obs->shape.size - 1) printf(", ");
        }
        printf("]\n");
        
        float reward = ((float*)step->reward->data)[0];
        printf("  Reward: %.2f, Done: %s\n", reward, step->done ? "Yes" : "No");
        
        // Cleanup
        cnp_rl_step_decref(step);
        cnp_tensor_decref(action);
        
        if (env->is_done) {
            printf("Episode finished!\n");
            break;
        }
    }
    
    // Cleanup
    cnp_rl_step_decref(initial_step);
    cnp_rl_env_decref(env);
    cnp_rl_cleanup();
    
    printf("\nCartPole test completed successfully!\n");
    return 0;
} 