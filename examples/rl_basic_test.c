#include "../src/cnumpy.h"
#include "../src/rl/cnumpy_rl.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("CNumpyRL - Basic Test\n");
    printf("====================\n\n");
    
    // Initialize
    cnp_rl_init();
    
    // Create environment
    cnp_rl_env_t *env = cnp_rl_gridworld_create(3, 3);
    printf("✓ Environment created\n");
    
    // Create agent
    cnp_rl_agent_t *agent = cnp_rl_random_agent_create("TestAgent", 4);
    printf("✓ Agent created\n");
    
    // Test single reset
    cnp_rl_step_t *step = env->reset(env);
    if (step) {
        printf("✓ Environment reset successful\n");
        cnp_rl_step_decref(step);
    } else {
        printf("✗ Environment reset failed\n");
        return 1;
    }
    
    // Test single step
    step = env->reset(env);
    cnp_tensor_t *action = agent->forward(agent, step);
    if (action) {
        printf("✓ Agent forward successful\n");
        
        cnp_rl_step_t *next_step = env->step(env, action);
        if (next_step) {
            printf("✓ Environment step successful\n");
            cnp_rl_step_decref(next_step);
        } else {
            printf("✗ Environment step failed\n");
        }
        
        cnp_tensor_decref(action);
    } else {
        printf("✗ Agent forward failed\n");
    }
    
    cnp_rl_step_decref(step);
    
    // Test trajectory creation
    printf("\nTesting trajectory creation...\n");
    cnp_rl_traj_t *traj = cnp_rl_traj_create(1);
    if (traj) {
        printf("✓ Trajectory created\n");
        cnp_rl_traj_decref(traj);
    } else {
        printf("✗ Trajectory creation failed\n");
    }
    
    // Test manual episode (without cnp_rl_run_episode)
    printf("\nTesting manual episode...\n");
    traj = cnp_rl_traj_create(1);
    cnp_rl_step_t *current_step = env->reset(env);
    
    for (int i = 0; i < 5 && current_step && !env->is_done; i++) {
        printf("  Step %d...\n", i + 1);
        
        cnp_tensor_t *act = agent->forward(agent, current_step);
        if (!act) {
            printf("    Agent forward failed\n");
            break;
        }
        
        cnp_rl_step_t *next = env->step(env, act);
        if (!next) {
            printf("    Environment step failed\n");
            cnp_tensor_decref(act);
            break;
        }
        
        printf("    ✓ Step %d completed\n", i + 1);
        
        // Add to trajectory
        cnp_rl_traj_add_step(traj, next);
        
        // Clean up
        cnp_rl_step_decref(current_step);
        current_step = next;
    }
    
    if (current_step) {
        cnp_rl_step_decref(current_step);
    }
    
    printf("✓ Manual episode completed with %zu steps\n", traj->num_steps);
    
    // Test reward calculation
    float reward = agent->reward(agent, traj);
    printf("✓ Total reward: %.2f\n", reward);
    
    cnp_rl_traj_decref(traj);
    
    // Cleanup
    cnp_rl_agent_decref(agent);
    cnp_rl_env_decref(env);
    cnp_rl_cleanup();
    
    printf("\n✓ Basic test completed successfully!\n");
    return 0;
} 