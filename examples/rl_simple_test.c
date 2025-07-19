#include "../cnumpy.h"
#include "../rl/cnumpy_rl.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("CNumpyRL - Simple Test\n");
    printf("=====================\n\n");
    
    // Initialize CNumpyRL
    cnp_rl_init();
    
    // Create a simple 3x3 grid world
    cnp_rl_env_t *env = cnp_rl_gridworld_create(3, 3);
    
    printf("Created %s environment\n", env->name);
    printf("Observation space: ");
    cnp_print_shape(&env->obs_shape);
    printf("\nAction space: %zu actions\n", env->num_actions);
    
    // Test environment reset
    printf("\nTesting environment reset...\n");
    cnp_rl_step_t *initial_step = env->reset(env);
    
    if (initial_step) {
        printf("Environment reset successful!\n");
        printf("Initial observation: ");
        cnp_print_tensor(initial_step->next_obs);
        cnp_rl_step_decref(initial_step);
    } else {
        printf("Environment reset failed!\n");
        return 1;
    }
    
    // Test random agent
    printf("\nTesting random agent...\n");
    cnp_rl_agent_t *agent = cnp_rl_random_agent_create("TestAgent", env->num_actions);
    
    if (agent) {
        printf("Random agent created successfully!\n");
        printf("Agent name: %s\n", agent->name);
    } else {
        printf("Random agent creation failed!\n");
        return 1;
    }
    
    // Test a few actions
    printf("\nTesting a few actions...\n");
    
    cnp_rl_step_t *current_step = env->reset(env);
    
    for (int i = 0; i < 5 && current_step && !env->is_done; i++) {
        printf("\nStep %d:\n", i + 1);
        
        // Agent chooses action
        cnp_tensor_t *action = agent->forward(agent, current_step);
        
        if (action) {
            float *action_data = (float*)action->data;
            printf("  Action: %d\n", (int)action_data[0]);
            
            // Environment steps
            cnp_rl_step_t *next_step = env->step(env, action);
            
            if (next_step) {
                printf("  Reward: ");
                cnp_print_tensor(next_step->reward);
                printf("  Done: %s\n", next_step->done ? "true" : "false");
                
                // Update current step
                cnp_rl_step_decref(current_step);
                current_step = next_step;
            } else {
                printf("  Environment step failed!\n");
                break;
            }
        } else {
            printf("  Agent forward failed!\n");
            break;
        }
    }
    
    // Test trajectory
    printf("\nTesting full episode...\n");
    cnp_rl_traj_t *traj = cnp_rl_run_episode(agent, env, 20, false);
    
    if (traj) {
        printf("Episode completed!\n");
        printf("Number of steps: %zu\n", traj->num_steps);
        
        float total_reward = agent->reward(agent, traj);
        printf("Total reward: %.2f\n", total_reward);
        
        cnp_rl_traj_decref(traj);
    } else {
        printf("Episode failed!\n");
    }
    
    // Cleanup
    printf("\nCleaning up...\n");
    
    if (current_step) {
        cnp_rl_step_decref(current_step);
    }
    
    cnp_rl_agent_decref(agent);
    cnp_rl_env_decref(env);
    
    cnp_rl_cleanup();
    
    printf("Simple RL test completed successfully!\n");
    
    return 0;
} 