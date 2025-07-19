#include <stdio.h>
#include <stdlib.h>
#include "../cnumpy.h"
#include "../rl/cnumpy_rl.h"

int main() {
    printf("Testing DQN Agent on CartPole...\n");
    
    // Initialize
    cnp_rl_init();
    srand(42);
    
    // Create CartPole environment
    cnp_rl_env_t *env = cnp_rl_cartpole_create();
    if (!env) {
        printf("Failed to create CartPole environment\n");
        return 1;
    }
    
    printf("Created CartPole environment: %s\n", env->name);
    
    // Create DQN agent
    cnp_rl_agent_t *agent = cnp_rl_dqn_agent_create(
        "DQN_Test",   // name
        4,            // observation size
        64,           // hidden size
        2,            // number of actions
        0.001f,       // learning rate
        1.0f,         // epsilon start
        0.01f,        // epsilon end
        0.995f,       // epsilon decay
        0.99f         // gamma
    );
    
    if (!agent) {
        printf("Failed to create DQN agent\n");
        cnp_rl_env_decref(env);
        return 1;
    }
    
    printf("Created DQN agent: %s\n", agent->name);
    
    // Test a few episodes
    printf("\nTesting training episodes...\n");
    for (int episode = 0; episode < 5; episode++) {
        printf("Episode %d:\n", episode);
        
        // Run episode
        cnp_rl_traj_t *traj = cnp_rl_run_episode(agent, env, 100, false);
        if (!traj) {
            printf("Failed to run episode\n");
            break;
        }
        
        // Calculate reward
        float episode_reward = agent->reward(agent, traj);
        printf("  Steps: %zu, Reward: %.2f\n", traj->num_steps, episode_reward);
        
        // Train agent
        agent->train(agent, traj);
        
        // Cleanup
        cnp_rl_traj_decref(traj);
    }
    
    // Cleanup
    cnp_rl_agent_decref(agent);
    cnp_rl_env_decref(env);
    cnp_rl_cleanup();
    
    printf("\nDQN CartPole test completed successfully!\n");
    return 0;
} 