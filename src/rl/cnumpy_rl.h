#ifndef CNUMPY_RL_H
#define CNUMPY_RL_H

#include "../cnumpy.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct cnp_rl_step cnp_rl_step_t;
typedef struct cnp_rl_traj cnp_rl_traj_t;
typedef struct cnp_rl_env cnp_rl_env_t;
typedef struct cnp_rl_agent cnp_rl_agent_t;
typedef struct cnp_rl_replay_buffer cnp_rl_replay_buffer_t;

// ============================================================================
// Core RL Data Structures
// ============================================================================

// Single step in the environment
struct cnp_rl_step {
    cnp_tensor_t *obs;          // Current observation
    cnp_tensor_t *next_obs;     // Next observation
    cnp_tensor_t *action;       // Action taken
    cnp_tensor_t *reward;       // Reward received
    bool done;                  // Episode terminal flag
    void *info;                 // Additional info (can be NULL)
    
    // Batch information
    size_t batch_size;          // Size of batch dimension (1 for single steps)
    
    // Memory management
    int ref_count;
};

// Trajectory (sequence of steps)
struct cnp_rl_traj {
    cnp_rl_step_t **steps;      // Array of steps
    size_t num_steps;           // Number of steps
    size_t capacity;            // Allocated capacity
    size_t batch_size;          // Batch size for all steps
    
    // Memory management
    int ref_count;
};

// ============================================================================
// Environment Interface
// ============================================================================

// Environment function pointers
typedef cnp_rl_step_t* (*cnp_rl_env_reset_fn_t)(cnp_rl_env_t *env);
typedef cnp_rl_step_t* (*cnp_rl_env_step_fn_t)(cnp_rl_env_t *env, cnp_tensor_t *action);
typedef void (*cnp_rl_env_render_fn_t)(cnp_rl_env_t *env);
typedef void (*cnp_rl_env_cleanup_fn_t)(cnp_rl_env_t *env);

// Base environment structure
struct cnp_rl_env {
    char *name;
    
    // Environment state
    cnp_tensor_t *current_obs;
    bool is_done;
    size_t step_count;
    
    // Observation and action spaces
    cnp_shape_t obs_shape;
    cnp_shape_t action_shape;
    size_t num_actions;         // For discrete action spaces
    
    // Batch information
    size_t batch_size;          // 1 for single environments
    
    // Function pointers
    cnp_rl_env_reset_fn_t reset;
    cnp_rl_env_step_fn_t step;
    cnp_rl_env_render_fn_t render;
    cnp_rl_env_cleanup_fn_t cleanup;
    
    // Environment-specific data
    void *env_data;
    
    // Memory management
    int ref_count;
};

// ============================================================================
// Agent Interface
// ============================================================================

// Agent function pointers
typedef cnp_tensor_t* (*cnp_rl_agent_forward_fn_t)(cnp_rl_agent_t *agent, cnp_rl_step_t *step);
typedef float (*cnp_rl_agent_reward_fn_t)(cnp_rl_agent_t *agent, cnp_rl_traj_t *traj);
typedef void (*cnp_rl_agent_train_fn_t)(cnp_rl_agent_t *agent, cnp_rl_traj_t *traj);
typedef void (*cnp_rl_agent_cleanup_fn_t)(cnp_rl_agent_t *agent);

// Base agent structure
struct cnp_rl_agent {
    char *name;
    
    // Agent state
    cnp_tensor_t **parameters;  // Trainable parameters
    size_t num_parameters;
    
    // Hyperparameters
    float learning_rate;
    float epsilon;              // For epsilon-greedy
    float gamma;                // Discount factor
    
    // Function pointers
    cnp_rl_agent_forward_fn_t forward;
    cnp_rl_agent_reward_fn_t reward;
    cnp_rl_agent_train_fn_t train;
    cnp_rl_agent_cleanup_fn_t cleanup;
    
    // Agent-specific data
    void *agent_data;
    
    // Memory management
    int ref_count;
};

// ============================================================================
// Replay Buffer
// ============================================================================

struct cnp_rl_replay_buffer {
    cnp_rl_step_t **buffer;     // Circular buffer of steps
    size_t capacity;            // Maximum buffer size
    size_t size;                // Current number of stored steps
    size_t head;                // Index of next write position
    
    // Sampling parameters
    size_t batch_size;          // Batch size for sampling
    bool sample_recent;         // Whether to sample recent experiences
    
    // Memory management
    int ref_count;
};

// ============================================================================
// Core API Functions
// ============================================================================

// RL system initialization
void cnp_rl_init(void);
void cnp_rl_cleanup(void);

// Step management
cnp_rl_step_t* cnp_rl_step_create(cnp_tensor_t *obs, cnp_tensor_t *next_obs,
                                  cnp_tensor_t *action, cnp_tensor_t *reward,
                                  bool done, void *info);
cnp_rl_step_t* cnp_rl_step_create_batch(size_t batch_size);
void cnp_rl_step_incref(cnp_rl_step_t *step);
void cnp_rl_step_decref(cnp_rl_step_t *step);
void cnp_rl_step_free(cnp_rl_step_t *step);

// Trajectory management
cnp_rl_traj_t* cnp_rl_traj_create(size_t batch_size);
void cnp_rl_traj_add_step(cnp_rl_traj_t *traj, cnp_rl_step_t *step);
void cnp_rl_traj_incref(cnp_rl_traj_t *traj);
void cnp_rl_traj_decref(cnp_rl_traj_t *traj);
void cnp_rl_traj_free(cnp_rl_traj_t *traj);

// Environment management
cnp_rl_env_t* cnp_rl_env_create(const char *name, const cnp_shape_t *obs_shape,
                                const cnp_shape_t *action_shape, size_t num_actions);
void cnp_rl_env_incref(cnp_rl_env_t *env);
void cnp_rl_env_decref(cnp_rl_env_t *env);
void cnp_rl_env_free(cnp_rl_env_t *env);

// Agent management
cnp_rl_agent_t* cnp_rl_agent_create(const char *name);
void cnp_rl_agent_incref(cnp_rl_agent_t *agent);
void cnp_rl_agent_decref(cnp_rl_agent_t *agent);
void cnp_rl_agent_free(cnp_rl_agent_t *agent);

// Replay buffer management
cnp_rl_replay_buffer_t* cnp_rl_replay_buffer_create(size_t capacity, size_t batch_size);
void cnp_rl_replay_buffer_add(cnp_rl_replay_buffer_t *buffer, cnp_rl_step_t *step);
cnp_rl_traj_t* cnp_rl_replay_buffer_sample(cnp_rl_replay_buffer_t *buffer);
void cnp_rl_replay_buffer_incref(cnp_rl_replay_buffer_t *buffer);
void cnp_rl_replay_buffer_decref(cnp_rl_replay_buffer_t *buffer);
void cnp_rl_replay_buffer_free(cnp_rl_replay_buffer_t *buffer);

// ============================================================================
// Specific Agent Implementations
// ============================================================================

// Random Agent
cnp_rl_agent_t* cnp_rl_random_agent_create(const char *name, size_t num_actions);

// DQN Agent
typedef struct {
    cnp_tensor_t **q_network;   // Q-network parameters
    cnp_tensor_t **target_network; // Target network parameters
    cnp_optimizer_t *optimizer;
    size_t network_size;
    size_t num_actions;          // Number of actions
    size_t target_update_freq;
    size_t update_count;
    float epsilon_start;
    float epsilon_end;
    float epsilon_decay;
} cnp_rl_dqn_data_t;

cnp_rl_agent_t* cnp_rl_dqn_agent_create(const char *name, size_t obs_size,
                                        size_t hidden_size, size_t num_actions,
                                        float learning_rate, float epsilon_start,
                                        float epsilon_end, float epsilon_decay,
                                        float gamma);

// ============================================================================
// Training Infrastructure
// ============================================================================

// Training loop
typedef struct {
    cnp_rl_agent_t *agent;
    cnp_rl_env_t *env;
    cnp_rl_replay_buffer_t *replay_buffer;
    size_t max_episodes;
    size_t max_steps_per_episode;
    size_t train_freq;          // How often to train
    size_t target_update_freq;  // How often to update target network
    bool render;
    bool verbose;
} cnp_rl_training_config_t;

typedef struct {
    float total_reward;
    float average_reward;
    size_t episode_length;
    size_t total_steps;
    float loss;                 // Training loss (if applicable)
} cnp_rl_training_stats_t;

// Training functions
cnp_rl_training_stats_t* cnp_rl_train_agent(cnp_rl_training_config_t *config);
cnp_rl_traj_t* cnp_rl_run_episode(cnp_rl_agent_t *agent, cnp_rl_env_t *env,
                                  size_t max_steps, bool render);

// ============================================================================
// Environment Implementations
// ============================================================================

// Simple grid world environment
cnp_rl_env_t* cnp_rl_gridworld_create(size_t width, size_t height);

// Classic control environments
cnp_rl_env_t* cnp_rl_cartpole_create(void);
cnp_rl_env_t* cnp_rl_mountaincar_create(void);

// Utility functions
void cnp_rl_print_step(const cnp_rl_step_t *step);
void cnp_rl_print_traj(const cnp_rl_traj_t *traj);
void cnp_rl_print_training_stats(const cnp_rl_training_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif // CNUMPY_RL_H 