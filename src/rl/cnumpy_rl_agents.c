#include "cnumpy_rl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ============================================================================
// Random Agent Implementation
// ============================================================================

typedef struct {
    size_t num_actions;
} cnp_rl_random_agent_data_t;

static cnp_tensor_t* cnp_rl_random_agent_forward(cnp_rl_agent_t *agent, cnp_rl_step_t *step) {
    cnp_rl_random_agent_data_t *data = (cnp_rl_random_agent_data_t*)agent->agent_data;
    
    // Choose random action
    int action = rand() % data->num_actions;
    
    // Create action tensor
    size_t action_dims[] = {1, 1};
    cnp_shape_t action_shape = cnp_shape_create(2, action_dims);
    
    float action_value = (float)action;
    cnp_tensor_t *action_tensor = cnp_tensor_create(&action_shape, CNP_FLOAT32, &action_value);
    
    cnp_shape_free(&action_shape);
    return action_tensor;
}

static float cnp_rl_random_agent_reward(cnp_rl_agent_t *agent, cnp_rl_traj_t *traj) {
    // Calculate total reward from trajectory
    float total_reward = 0.0f;
    
    for (size_t i = 0; i < traj->num_steps; i++) {
        cnp_rl_step_t *step = traj->steps[i];
        if (step->reward) {
            float *reward_data = (float*)step->reward->data;
            total_reward += reward_data[0];
        }
    }
    
    return total_reward;
}

static void cnp_rl_random_agent_train(cnp_rl_agent_t *agent, cnp_rl_traj_t *traj) {
    // Random agent doesn't learn, so nothing to do
    (void)agent;
    (void)traj;
}

static void cnp_rl_random_agent_cleanup(cnp_rl_agent_t *agent) {
    if (agent->agent_data) {
        free(agent->agent_data);
        agent->agent_data = NULL;
    }
}

cnp_rl_agent_t* cnp_rl_random_agent_create(const char *name, size_t num_actions) {
    cnp_rl_agent_t *agent = cnp_rl_agent_create(name);
    
    // Create random agent specific data
    cnp_rl_random_agent_data_t *data = malloc(sizeof(cnp_rl_random_agent_data_t));
    data->num_actions = num_actions;
    
    // Set function pointers
    agent->forward = cnp_rl_random_agent_forward;
    agent->reward = cnp_rl_random_agent_reward;
    agent->train = cnp_rl_random_agent_train;
    agent->cleanup = cnp_rl_random_agent_cleanup;
    
    // Set agent data
    agent->agent_data = data;
    
    return agent;
}

// ============================================================================
// DQN Agent Implementation
// ============================================================================

// Helper function to create simple neural network for DQN
static cnp_tensor_t** cnp_rl_create_dqn_network(size_t obs_size, size_t hidden_size, size_t num_actions, size_t *network_size) {
    // Simple 2-layer network: input -> hidden -> output
    // Parameters: W1, b1, W2, b2
    
    *network_size = 4;
    cnp_tensor_t **network = malloc(sizeof(cnp_tensor_t*) * 4);
    
    // Layer 1: input -> hidden
    size_t w1_dims[] = {obs_size, hidden_size};
    cnp_shape_t w1_shape = cnp_shape_create(2, w1_dims);
    network[0] = cnp_uniform(&w1_shape, CNP_FLOAT32, -0.1f, 0.1f);
    network[0]->requires_grad = true;
    
    size_t b1_dims[] = {1, hidden_size};
    cnp_shape_t b1_shape = cnp_shape_create(2, b1_dims);
    network[1] = cnp_zeros(&b1_shape, CNP_FLOAT32);
    network[1]->requires_grad = true;
    
    // Layer 2: hidden -> output
    size_t w2_dims[] = {hidden_size, num_actions};
    cnp_shape_t w2_shape = cnp_shape_create(2, w2_dims);
    network[2] = cnp_uniform(&w2_shape, CNP_FLOAT32, -0.1f, 0.1f);
    network[2]->requires_grad = true;
    
    size_t b2_dims[] = {1, num_actions};
    cnp_shape_t b2_shape = cnp_shape_create(2, b2_dims);
    network[3] = cnp_zeros(&b2_shape, CNP_FLOAT32);
    network[3]->requires_grad = true;
    
    cnp_shape_free(&w1_shape);
    cnp_shape_free(&b1_shape);
    cnp_shape_free(&w2_shape);
    cnp_shape_free(&b2_shape);
    
    return network;
}

// Forward pass through DQN network
static cnp_tensor_t* cnp_rl_dqn_forward_network(cnp_tensor_t **network, cnp_tensor_t *input) {
    // Layer 1: input @ W1 + b1
    cnp_tensor_t *z1 = cnp_matmul(input, network[0]);
    cnp_tensor_t *a1 = cnp_add(z1, network[1]);
    cnp_tensor_t *h1 = cnp_relu(a1);
    
    // Layer 2: h1 @ W2 + b2
    cnp_tensor_t *z2 = cnp_matmul(h1, network[2]);
    cnp_tensor_t *output = cnp_add(z2, network[3]);
    
    return output;
}

static cnp_tensor_t* cnp_rl_dqn_agent_forward(cnp_rl_agent_t *agent, cnp_rl_step_t *step) {
    cnp_rl_dqn_data_t *data = (cnp_rl_dqn_data_t*)agent->agent_data;
    
    // Get Q-values from network (use current obs, not next_obs)
    cnp_tensor_t *observation = step->next_obs ? step->next_obs : step->obs;
    
    // Reshape observation to 2D if needed (for matrix multiplication)
    cnp_tensor_t *obs_input = observation;
    if (observation->shape.ndim == 1) {
        size_t new_dims[] = {1, observation->shape.size};
        cnp_shape_t new_shape = cnp_shape_create(2, new_dims);
        obs_input = cnp_reshape(observation, &new_shape);
        cnp_shape_free(&new_shape);
    }
    
    cnp_tensor_t *q_values = cnp_rl_dqn_forward_network(data->q_network, obs_input);
    
    // Cleanup reshaped tensor if we created one
    if (obs_input != observation) {
        cnp_tensor_decref(obs_input);
    }
    
    // Epsilon-greedy action selection
    float epsilon = data->epsilon_start + (data->epsilon_end - data->epsilon_start) * 
                   expf(-data->update_count / data->epsilon_decay);
    
    int action;
    if ((float)rand() / RAND_MAX < epsilon) {
        // Random action
        action = rand() % data->num_actions;
    } else {
        // Greedy action (argmax of Q-values)
        float *q_data = (float*)q_values->data;
        action = 0;
        float max_q = q_data[0];
        for (size_t i = 1; i < q_values->shape.dims[1]; i++) {
            if (q_data[i] > max_q) {
                max_q = q_data[i];
                action = i;
            }
        }
    }
    
    // Create action tensor
    size_t action_dims[] = {1};
    cnp_shape_t action_shape = cnp_shape_create(1, action_dims);
    
    float action_value = (float)action;
    cnp_tensor_t *action_tensor = cnp_tensor_create(&action_shape, CNP_FLOAT32, &action_value);
    
    cnp_shape_free(&action_shape);
    return action_tensor;
}

static float cnp_rl_dqn_agent_reward(cnp_rl_agent_t *agent, cnp_rl_traj_t *traj) {
    // Calculate total reward from trajectory
    float total_reward = 0.0f;
    
    for (size_t i = 0; i < traj->num_steps; i++) {
        cnp_rl_step_t *step = traj->steps[i];
        if (step->reward) {
            float *reward_data = (float*)step->reward->data;
            total_reward += reward_data[0];
        }
    }
    
    return total_reward;
}

static void cnp_rl_dqn_agent_train(cnp_rl_agent_t *agent, cnp_rl_traj_t *traj) {
    cnp_rl_dqn_data_t *data = (cnp_rl_dqn_data_t*)agent->agent_data;
    
    // Simple DQN training on trajectory
    for (size_t i = 0; i < traj->num_steps; i++) {
        cnp_rl_step_t *step = traj->steps[i];
        
        if (!step->obs || !step->next_obs || !step->action || !step->reward) continue;
        
        // Reshape observations if needed
        cnp_tensor_t *obs_input = step->obs;
        if (step->obs->shape.ndim == 1) {
            size_t new_dims[] = {1, step->obs->shape.size};
            cnp_shape_t new_shape = cnp_shape_create(2, new_dims);
            obs_input = cnp_reshape(step->obs, &new_shape);
            cnp_shape_free(&new_shape);
        }
        
        cnp_tensor_t *next_obs_input = step->next_obs;
        if (step->next_obs->shape.ndim == 1) {
            size_t new_dims[] = {1, step->next_obs->shape.size};
            cnp_shape_t new_shape = cnp_shape_create(2, new_dims);
            next_obs_input = cnp_reshape(step->next_obs, &new_shape);
            cnp_shape_free(&new_shape);
        }
        
        // Get current Q-values
        cnp_tensor_t *q_values = cnp_rl_dqn_forward_network(data->q_network, obs_input);
        
        // Get target Q-values
        cnp_tensor_t *next_q_values = cnp_rl_dqn_forward_network(data->target_network, next_obs_input);
        
        // Calculate target
        float *reward_data = (float*)step->reward->data;
        float reward = reward_data[0];
        
        float target = reward;
        if (!step->done) {
            // Find max Q-value for next state
            float *next_q_data = (float*)next_q_values->data;
            float max_next_q = next_q_data[0];
            for (size_t j = 1; j < next_q_values->shape.dims[1]; j++) {
                if (next_q_data[j] > max_next_q) {
                    max_next_q = next_q_data[j];
                }
            }
            target += agent->gamma * max_next_q;
        }
        
        // Create target tensor
        size_t target_dims[] = {1, q_values->shape.dims[1]};
        cnp_shape_t target_shape = cnp_shape_create(2, target_dims);
        cnp_tensor_t *target_tensor = cnp_tensor_alloc(&target_shape, CNP_FLOAT32);
        
        // Copy Q-values and update the action that was taken
        float *q_data = (float*)q_values->data;
        float *target_data = (float*)target_tensor->data;
        
        for (size_t j = 0; j < q_values->shape.dims[1]; j++) {
            target_data[j] = q_data[j];
        }
        
        // Update target for the action that was taken
        float *action_data = (float*)step->action->data;
        int action = (int)action_data[0];
        target_data[action] = target;
        
        // Compute loss (mean squared error)
        cnp_tensor_t *diff = cnp_sub(q_values, target_tensor);
        cnp_tensor_t *loss = cnp_mul(diff, diff);
        cnp_tensor_t *mean_loss = cnp_reduce_sum(loss, -1);
        
        // Backward pass
        if (!mean_loss->grad) {
            mean_loss->grad = cnp_ones(&mean_loss->shape, CNP_FLOAT32);
        }
        cnp_backward(mean_loss);
        
        // Update parameters
        data->optimizer->minimize(data->optimizer, mean_loss);
        
        // Zero gradients
        for (size_t j = 0; j < data->network_size; j++) {
            cnp_zero_grad(data->q_network[j]);
        }
        
        cnp_shape_free(&target_shape);
        
        // Cleanup reshaped tensors
        if (obs_input != step->obs) {
            cnp_tensor_decref(obs_input);
        }
        if (next_obs_input != step->next_obs) {
            cnp_tensor_decref(next_obs_input);
        }
        
        data->update_count++;
    }
    
    // Update target network periodically
    if (data->update_count % data->target_update_freq == 0) {
        // Copy Q-network to target network
        for (size_t i = 0; i < data->network_size; i++) {
            memcpy(data->target_network[i]->data, data->q_network[i]->data,
                   cnp_dtype_size(data->q_network[i]->dtype) * data->q_network[i]->shape.size);
        }
    }
}

static void cnp_rl_dqn_agent_cleanup(cnp_rl_agent_t *agent) {
    cnp_rl_dqn_data_t *data = (cnp_rl_dqn_data_t*)agent->agent_data;
    
    if (data) {
        // Free networks
        if (data->q_network) {
            for (size_t i = 0; i < data->network_size; i++) {
                cnp_tensor_decref(data->q_network[i]);
            }
            free(data->q_network);
        }
        
        if (data->target_network) {
            for (size_t i = 0; i < data->network_size; i++) {
                cnp_tensor_decref(data->target_network[i]);
            }
            free(data->target_network);
        }
        
        // Free optimizer
        if (data->optimizer) {
            cnp_optimizer_free(data->optimizer);
        }
        
        free(data);
        agent->agent_data = NULL;
    }
}

cnp_rl_agent_t* cnp_rl_dqn_agent_create(const char *name, size_t obs_size,
                                        size_t hidden_size, size_t num_actions,
                                        float learning_rate, float epsilon_start,
                                        float epsilon_end, float epsilon_decay,
                                        float gamma) {
    cnp_rl_agent_t *agent = cnp_rl_agent_create(name);
    
    // Set hyperparameters
    agent->learning_rate = learning_rate;
    agent->gamma = gamma;
    
    // Create DQN specific data
    cnp_rl_dqn_data_t *data = malloc(sizeof(cnp_rl_dqn_data_t));
    
    // Create Q-network
    data->q_network = cnp_rl_create_dqn_network(obs_size, hidden_size, num_actions, &data->network_size);
    
    // Create target network (copy of Q-network)
    data->target_network = cnp_rl_create_dqn_network(obs_size, hidden_size, num_actions, &data->network_size);
    
    // Copy Q-network weights to target network
    for (size_t i = 0; i < data->network_size; i++) {
        memcpy(data->target_network[i]->data, data->q_network[i]->data,
               cnp_dtype_size(data->q_network[i]->dtype) * data->q_network[i]->shape.size);
    }
    
    // Create optimizer
    data->optimizer = cnp_sgd_create(learning_rate, false);
    
    // Set parameters
    data->num_actions = num_actions;
    data->target_update_freq = 100;  // Update target network every 100 steps
    data->update_count = 0;
    data->epsilon_start = epsilon_start;
    data->epsilon_end = epsilon_end;
    data->epsilon_decay = epsilon_decay;
    
    // Set function pointers
    agent->forward = cnp_rl_dqn_agent_forward;
    agent->reward = cnp_rl_dqn_agent_reward;
    agent->train = cnp_rl_dqn_agent_train;
    agent->cleanup = cnp_rl_dqn_agent_cleanup;
    
    // Set agent data
    agent->agent_data = data;
    
    return agent;
} 