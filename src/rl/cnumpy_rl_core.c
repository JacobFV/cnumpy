#include "cnumpy_rl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

// Global state for RL
static bool g_rl_initialized = false;

// ============================================================================
// RL System Initialization
// ============================================================================

void cnp_rl_init(void) {
    if (g_rl_initialized) return;
    
    // Initialize main cnumpy if not already done
    cnp_init();
    
    // Initialize random seed for RL
    srand(time(NULL));
    
    g_rl_initialized = true;
}

void cnp_rl_cleanup(void) {
    if (!g_rl_initialized) return;
    
    // Cleanup will be handled by individual object cleanup
    g_rl_initialized = false;
}

// ============================================================================
// Step Management
// ============================================================================

cnp_rl_step_t* cnp_rl_step_create(cnp_tensor_t *obs, cnp_tensor_t *next_obs,
                                  cnp_tensor_t *action, cnp_tensor_t *reward,
                                  bool done, void *info) {
    cnp_rl_step_t *step = malloc(sizeof(cnp_rl_step_t));
    
    step->obs = obs;
    step->next_obs = next_obs;
    step->action = action;
    step->reward = reward;
    step->done = done;
    step->info = info;
    
    // For single environment interactions, batch size is always 1
    step->batch_size = 1;
    
    step->ref_count = 1;
    
    // Increment reference counts for tensors
    if (obs) cnp_tensor_incref(obs);
    if (next_obs) cnp_tensor_incref(next_obs);
    if (action) cnp_tensor_incref(action);
    if (reward) cnp_tensor_incref(reward);
    
    return step;
}

cnp_rl_step_t* cnp_rl_step_create_batch(size_t batch_size) {
    cnp_rl_step_t *step = malloc(sizeof(cnp_rl_step_t));
    
    step->obs = NULL;
    step->next_obs = NULL;
    step->action = NULL;
    step->reward = NULL;
    step->done = false;
    step->info = NULL;
    step->batch_size = batch_size;
    step->ref_count = 1;
    
    return step;
}

void cnp_rl_step_incref(cnp_rl_step_t *step) {
    if (step) {
        step->ref_count++;
    }
}

void cnp_rl_step_decref(cnp_rl_step_t *step) {
    if (!step) return;
    
    step->ref_count--;
    if (step->ref_count <= 0) {
        cnp_rl_step_free(step);
    }
}

void cnp_rl_step_free(cnp_rl_step_t *step) {
    if (!step) return;
    
    // Decrement reference counts for tensors
    if (step->obs) cnp_tensor_decref(step->obs);
    if (step->next_obs) cnp_tensor_decref(step->next_obs);
    if (step->action) cnp_tensor_decref(step->action);
    if (step->reward) cnp_tensor_decref(step->reward);
    
    // Note: info is not freed as it's managed by the caller
    
    free(step);
}

// ============================================================================
// Trajectory Management
// ============================================================================

cnp_rl_traj_t* cnp_rl_traj_create(size_t batch_size) {
    cnp_rl_traj_t *traj = malloc(sizeof(cnp_rl_traj_t));
    
    traj->steps = NULL;
    traj->num_steps = 0;
    traj->capacity = 0;
    traj->batch_size = batch_size;
    traj->ref_count = 1;
    
    return traj;
}

void cnp_rl_traj_add_step(cnp_rl_traj_t *traj, cnp_rl_step_t *step) {
    if (!traj || !step) return;
    
    // Check batch size compatibility
    if (traj->batch_size != step->batch_size) {
        fprintf(stderr, "Warning: Batch size mismatch in trajectory (%zu vs %zu)\n",
                traj->batch_size, step->batch_size);
    }
    
    // Expand capacity if needed
    if (traj->num_steps >= traj->capacity) {
        traj->capacity = traj->capacity == 0 ? 8 : traj->capacity * 2;
        traj->steps = realloc(traj->steps, sizeof(cnp_rl_step_t*) * traj->capacity);
    }
    
    // Add step and increment its reference count
    traj->steps[traj->num_steps] = step;
    cnp_rl_step_incref(step);
    traj->num_steps++;
}

void cnp_rl_traj_incref(cnp_rl_traj_t *traj) {
    if (traj) {
        traj->ref_count++;
    }
}

void cnp_rl_traj_decref(cnp_rl_traj_t *traj) {
    if (!traj) return;
    
    traj->ref_count--;
    if (traj->ref_count <= 0) {
        cnp_rl_traj_free(traj);
    }
}

void cnp_rl_traj_free(cnp_rl_traj_t *traj) {
    if (!traj) return;
    
    // Decrement reference counts for all steps
    for (size_t i = 0; i < traj->num_steps; i++) {
        cnp_rl_step_decref(traj->steps[i]);
    }
    
    // Free the steps array
    if (traj->steps) {
        free(traj->steps);
    }
    
    free(traj);
}

// ============================================================================
// Environment Management
// ============================================================================

cnp_rl_env_t* cnp_rl_env_create(const char *name, const cnp_shape_t *obs_shape,
                                const cnp_shape_t *action_shape, size_t num_actions) {
    cnp_rl_env_t *env = malloc(sizeof(cnp_rl_env_t));
    
    // Set name
    env->name = malloc(strlen(name) + 1);
    strcpy(env->name, name);
    
    // Initialize state
    env->current_obs = NULL;
    env->is_done = false;
    env->step_count = 0;
    
    // Copy shapes
    env->obs_shape = cnp_shape_copy(obs_shape);
    env->action_shape = cnp_shape_copy(action_shape);
    env->num_actions = num_actions;
    
    // Default to single environment
    env->batch_size = 1;
    
    // Initialize function pointers to NULL
    env->reset = NULL;
    env->step = NULL;
    env->render = NULL;
    env->cleanup = NULL;
    
    env->env_data = NULL;
    env->ref_count = 1;
    
    return env;
}

void cnp_rl_env_incref(cnp_rl_env_t *env) {
    if (env) {
        env->ref_count++;
    }
}

void cnp_rl_env_decref(cnp_rl_env_t *env) {
    if (!env) return;
    
    env->ref_count--;
    if (env->ref_count <= 0) {
        cnp_rl_env_free(env);
    }
}

void cnp_rl_env_free(cnp_rl_env_t *env) {
    if (!env) return;
    
    // Call cleanup if available
    if (env->cleanup) {
        env->cleanup(env);
    }
    
    // Free name
    if (env->name) {
        free(env->name);
    }
    
    // Free current observation
    if (env->current_obs) {
        cnp_tensor_decref(env->current_obs);
    }
    
    // Free shapes
    cnp_shape_free(&env->obs_shape);
    cnp_shape_free(&env->action_shape);
    
    free(env);
}

// ============================================================================
// Agent Management
// ============================================================================

cnp_rl_agent_t* cnp_rl_agent_create(const char *name) {
    cnp_rl_agent_t *agent = malloc(sizeof(cnp_rl_agent_t));
    
    // Set name
    agent->name = malloc(strlen(name) + 1);
    strcpy(agent->name, name);
    
    // Initialize state
    agent->parameters = NULL;
    agent->num_parameters = 0;
    
    // Default hyperparameters
    agent->learning_rate = 0.001f;
    agent->epsilon = 0.1f;
    agent->gamma = 0.99f;
    
    // Initialize function pointers to NULL
    agent->forward = NULL;
    agent->reward = NULL;
    agent->train = NULL;
    agent->cleanup = NULL;
    
    agent->agent_data = NULL;
    agent->ref_count = 1;
    
    return agent;
}

void cnp_rl_agent_incref(cnp_rl_agent_t *agent) {
    if (agent) {
        agent->ref_count++;
    }
}

void cnp_rl_agent_decref(cnp_rl_agent_t *agent) {
    if (!agent) return;
    
    agent->ref_count--;
    if (agent->ref_count <= 0) {
        cnp_rl_agent_free(agent);
    }
}

void cnp_rl_agent_free(cnp_rl_agent_t *agent) {
    if (!agent) return;
    
    // Call cleanup if available
    if (agent->cleanup) {
        agent->cleanup(agent);
    }
    
    // Free name
    if (agent->name) {
        free(agent->name);
    }
    
    // Free parameters
    if (agent->parameters) {
        for (size_t i = 0; i < agent->num_parameters; i++) {
            cnp_tensor_decref(agent->parameters[i]);
        }
        free(agent->parameters);
    }
    
    free(agent);
}

// ============================================================================
// Replay Buffer Management
// ============================================================================

cnp_rl_replay_buffer_t* cnp_rl_replay_buffer_create(size_t capacity, size_t batch_size) {
    cnp_rl_replay_buffer_t *buffer = malloc(sizeof(cnp_rl_replay_buffer_t));
    
    buffer->buffer = malloc(sizeof(cnp_rl_step_t*) * capacity);
    buffer->capacity = capacity;
    buffer->size = 0;
    buffer->head = 0;
    buffer->batch_size = batch_size;
    buffer->sample_recent = false;
    buffer->ref_count = 1;
    
    // Initialize buffer to NULL
    for (size_t i = 0; i < capacity; i++) {
        buffer->buffer[i] = NULL;
    }
    
    return buffer;
}

void cnp_rl_replay_buffer_add(cnp_rl_replay_buffer_t *buffer, cnp_rl_step_t *step) {
    if (!buffer || !step) return;
    
    // If buffer is full, decrement reference count of step being replaced
    if (buffer->size == buffer->capacity && buffer->buffer[buffer->head]) {
        cnp_rl_step_decref(buffer->buffer[buffer->head]);
    }
    
    // Add new step
    buffer->buffer[buffer->head] = step;
    cnp_rl_step_incref(step);
    
    // Update head position
    buffer->head = (buffer->head + 1) % buffer->capacity;
    
    // Update size
    if (buffer->size < buffer->capacity) {
        buffer->size++;
    }
}

cnp_rl_traj_t* cnp_rl_replay_buffer_sample(cnp_rl_replay_buffer_t *buffer) {
    if (!buffer || buffer->size == 0) return NULL;
    
    cnp_rl_traj_t *traj = cnp_rl_traj_create(buffer->batch_size);
    
    // Sample random steps
    for (size_t i = 0; i < buffer->batch_size && i < buffer->size; i++) {
        size_t idx = rand() % buffer->size;
        cnp_rl_step_t *step = buffer->buffer[idx];
        
        if (step) {
            cnp_rl_traj_add_step(traj, step);
        }
    }
    
    return traj;
}

void cnp_rl_replay_buffer_incref(cnp_rl_replay_buffer_t *buffer) {
    if (buffer) {
        buffer->ref_count++;
    }
}

void cnp_rl_replay_buffer_decref(cnp_rl_replay_buffer_t *buffer) {
    if (!buffer) return;
    
    buffer->ref_count--;
    if (buffer->ref_count <= 0) {
        cnp_rl_replay_buffer_free(buffer);
    }
}

void cnp_rl_replay_buffer_free(cnp_rl_replay_buffer_t *buffer) {
    if (!buffer) return;
    
    // Decrement reference counts for all stored steps
    for (size_t i = 0; i < buffer->size; i++) {
        if (buffer->buffer[i]) {
            cnp_rl_step_decref(buffer->buffer[i]);
        }
    }
    
    // Free the buffer array
    if (buffer->buffer) {
        free(buffer->buffer);
    }
    
    free(buffer);
}

// ============================================================================
// Utility Functions
// ============================================================================

void cnp_rl_print_step(const cnp_rl_step_t *step) {
    if (!step) {
        printf("Step: NULL\n");
        return;
    }
    
    printf("Step(batch_size=%zu, done=%s):\n", step->batch_size, step->done ? "true" : "false");
    
    if (step->obs) {
        printf("  obs: ");
        cnp_print_tensor(step->obs);
    }
    
    if (step->next_obs) {
        printf("  next_obs: ");
        cnp_print_tensor(step->next_obs);
    }
    
    if (step->action) {
        printf("  action: ");
        cnp_print_tensor(step->action);
    }
    
    if (step->reward) {
        printf("  reward: ");
        cnp_print_tensor(step->reward);
    }
}

void cnp_rl_print_traj(const cnp_rl_traj_t *traj) {
    if (!traj) {
        printf("Trajectory: NULL\n");
        return;
    }
    
    printf("Trajectory(batch_size=%zu, num_steps=%zu):\n", traj->batch_size, traj->num_steps);
    
    for (size_t i = 0; i < traj->num_steps; i++) {
        printf("  Step %zu:\n", i);
        cnp_rl_print_step(traj->steps[i]);
    }
}

void cnp_rl_print_training_stats(const cnp_rl_training_stats_t *stats) {
    if (!stats) {
        printf("Training Stats: NULL\n");
        return;
    }
    
    printf("Training Stats:\n");
    printf("  Total Reward: %.6f\n", stats->total_reward);
    printf("  Average Reward: %.6f\n", stats->average_reward);
    printf("  Episode Length: %zu\n", stats->episode_length);
    printf("  Total Steps: %zu\n", stats->total_steps);
    printf("  Loss: %.6f\n", stats->loss);
} 