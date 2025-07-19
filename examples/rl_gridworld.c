#include "../cnumpy.h"
#include "../rl/cnumpy_rl.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("CNumpyRL - Grid World Example\n");
    printf("=============================\n\n");
    
    // Initialize CNumpyRL
    cnp_rl_init();
    
    // Create a simple 4x4 grid world
    cnp_rl_env_t *env = cnp_rl_gridworld_create(4, 4);
    
    printf("Created %s environment (%zu x %zu)\n", env->name, 4, 4);
    printf("Observation space: ");
    cnp_print_shape(&env->obs_shape);
    printf("\nAction space: %zu actions\n", env->num_actions);
    printf("Actions: 0=up, 1=right, 2=down, 3=left\n\n");
    
    // ====================================================================
    // Test Random Agent
    // ====================================================================
    
    printf("Testing Random Agent\n");
    printf("===================\n");
    
    // Create random agent
    cnp_rl_agent_t *random_agent = cnp_rl_random_agent_create("RandomAgent", env->num_actions);
    
    // Run a few episodes with random agent
    printf("Running 3 episodes with random agent...\n");
    
    for (int episode = 0; episode < 3; episode++) {
        printf("\n--- Episode %d ---\n", episode + 1);
        
        cnp_rl_traj_t *traj = cnp_rl_run_episode(random_agent, env, 50, false);
        
        if (traj) {
            float total_reward = random_agent->reward(random_agent, traj);
            printf("Episode %d: Steps=%zu, Total Reward=%.2f\n", 
                   episode + 1, traj->num_steps, total_reward);
            
            cnp_rl_traj_decref(traj);
        }
    }
    
    // ====================================================================
    // Test DQN Agent
    // ====================================================================
    
    printf("\n\nTesting DQN Agent\n");
    printf("================\n");
    
    // Create DQN agent
    size_t obs_size = env->obs_shape.size;  // Flattened observation space
    size_t hidden_size = 16;
    size_t num_actions = env->num_actions;
    
    cnp_rl_agent_t *dqn_agent = cnp_rl_dqn_agent_create(
        "DQNAgent", obs_size, hidden_size, num_actions,
        0.01f,    // learning_rate
        1.0f,     // epsilon_start
        0.1f,     // epsilon_end
        200.0f,   // epsilon_decay
        0.99f     // gamma
    );
    
    printf("Created DQN agent with %zu observations -> %zu hidden -> %zu actions\n",
           obs_size, hidden_size, num_actions);
    
    // Create replay buffer
    cnp_rl_replay_buffer_t *replay_buffer = cnp_rl_replay_buffer_create(1000, 32);
    
    // Training configuration
    cnp_rl_training_config_t config = {
        .agent = dqn_agent,
        .env = env,
        .replay_buffer = replay_buffer,
        .max_episodes = 100,
        .max_steps_per_episode = 50,
        .train_freq = 1,
        .target_update_freq = 10,
        .render = false,
        .verbose = true
    };
    
    printf("\nTraining DQN agent for %zu episodes...\n", config.max_episodes);
    
    // Train the agent
    cnp_rl_training_stats_t *stats = cnp_rl_train_agent(&config);
    
    if (stats) {
        printf("\nTraining completed!\n");
        cnp_rl_print_training_stats(stats);
        free(stats);
    }
    
    // ====================================================================
    // Test Trained Agent
    // ====================================================================
    
    printf("\n\nTesting Trained DQN Agent\n");
    printf("=========================\n");
    
    // Run a few episodes with trained agent (with rendering)
    printf("Running 3 episodes with trained agent...\n");
    
    for (int episode = 0; episode < 3; episode++) {
        printf("\n--- Trained Episode %d ---\n", episode + 1);
        
        cnp_rl_traj_t *traj = cnp_rl_run_episode(dqn_agent, env, 50, true);
        
        if (traj) {
            float total_reward = dqn_agent->reward(dqn_agent, traj);
            printf("Trained Episode %d: Steps=%zu, Total Reward=%.2f\n", 
                   episode + 1, traj->num_steps, total_reward);
            
            cnp_rl_traj_decref(traj);
        }
    }
    
    // ====================================================================
    // Performance Comparison
    // ====================================================================
    
    printf("\n\nPerformance Comparison\n");
    printf("=====================\n");
    
    // Run multiple episodes to compare performance
    int num_test_episodes = 10;
    float random_rewards[num_test_episodes];
    float dqn_rewards[num_test_episodes];
    
    printf("Running %d test episodes for comparison...\n", num_test_episodes);
    
    // Test random agent
    for (int i = 0; i < num_test_episodes; i++) {
        cnp_rl_traj_t *traj = cnp_rl_run_episode(random_agent, env, 50, false);
        random_rewards[i] = traj ? random_agent->reward(random_agent, traj) : 0.0f;
        if (traj) cnp_rl_traj_decref(traj);
    }
    
    // Test DQN agent
    for (int i = 0; i < num_test_episodes; i++) {
        cnp_rl_traj_t *traj = cnp_rl_run_episode(dqn_agent, env, 50, false);
        dqn_rewards[i] = traj ? dqn_agent->reward(dqn_agent, traj) : 0.0f;
        if (traj) cnp_rl_traj_decref(traj);
    }
    
    // Calculate averages
    float random_avg = 0.0f, dqn_avg = 0.0f;
    for (int i = 0; i < num_test_episodes; i++) {
        random_avg += random_rewards[i];
        dqn_avg += dqn_rewards[i];
    }
    random_avg /= num_test_episodes;
    dqn_avg /= num_test_episodes;
    
    printf("\nResults:\n");
    printf("Random Agent Average Reward: %.2f\n", random_avg);
    printf("DQN Agent Average Reward: %.2f\n", dqn_avg);
    printf("Improvement: %.2f%%\n", 
           random_avg != 0 ? ((dqn_avg - random_avg) / random_avg) * 100 : 0);
    
    // ====================================================================
    // Cleanup
    // ====================================================================
    
    printf("\nCleaning up resources...\n");
    
    // Free agents
    cnp_rl_agent_decref(random_agent);
    cnp_rl_agent_decref(dqn_agent);
    
    // Free environment
    cnp_rl_env_decref(env);
    
    // Free replay buffer
    cnp_rl_replay_buffer_decref(replay_buffer);
    
    // Cleanup RL system
    cnp_rl_cleanup();
    
    printf("\nGrid World RL example completed successfully!\n");
    
    return 0;
} 