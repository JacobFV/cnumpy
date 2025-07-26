#include "cnumpy_rl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <assert.h>

// ============================================================================
// Grid World Environment Implementation
// ============================================================================

typedef struct {
    size_t width;
    size_t height;
    size_t agent_x;
    size_t agent_y;
    size_t goal_x;
    size_t goal_y;
    float reward_goal;
    float reward_step;
    float reward_wall;
    size_t max_steps;
    bool goal_reached;
} cnp_rl_gridworld_data_t;

// Helper function to create observation tensor for grid world
static cnp_tensor_t* cnp_rl_gridworld_create_obs(cnp_rl_gridworld_data_t *data) {
    // Create observation as a one-hot encoded position
    // Observation shape: [width * height]
    size_t obs_dims[] = {1, data->width * data->height};
    cnp_shape_t obs_shape = cnp_shape_create(2, obs_dims);
    
    cnp_tensor_t *obs = cnp_zeros(&obs_shape, CNP_FLOAT32);
    
    // Set the agent position to 1
    size_t agent_pos = data->agent_y * data->width + data->agent_x;
    float *obs_data = (float*)obs->data;
    obs_data[agent_pos] = 1.0f;
    
    cnp_shape_free(&obs_shape);
    return obs;
}

static cnp_rl_step_t* cnp_rl_gridworld_reset(cnp_rl_env_t *env) {
    cnp_rl_gridworld_data_t *data = (cnp_rl_gridworld_data_t*)env->env_data;
    
    // Reset agent position to (0, 0)
    data->agent_x = 0;
    data->agent_y = 0;
    
    // Reset goal position to opposite corner
    data->goal_x = data->width - 1;
    data->goal_y = data->height - 1;
    
    // Reset episode state
    env->step_count = 0;
    env->is_done = false;
    data->goal_reached = false;
    
    // Create initial observation
    cnp_tensor_t *obs = cnp_rl_gridworld_create_obs(data);
    
    // Update environment's current observation
    if (env->current_obs) {
        cnp_tensor_decref(env->current_obs);
    }
    env->current_obs = obs;
    cnp_tensor_incref(obs);
    
    // Create initial step
    cnp_rl_step_t *step = cnp_rl_step_create(NULL, obs, NULL, NULL, false, NULL);
    
    return step;
}

static cnp_rl_step_t* cnp_rl_gridworld_step(cnp_rl_env_t *env, cnp_tensor_t *action) {
    cnp_rl_gridworld_data_t *data = (cnp_rl_gridworld_data_t*)env->env_data;
    
    // Get action value (0=up, 1=right, 2=down, 3=left)
    float *action_data = (float*)action->data;
    int action_val = (int)action_data[0];
    
    // Store previous observation
    cnp_tensor_t *prev_obs = env->current_obs;
    cnp_tensor_incref(prev_obs);
    
    // Store previous position
    size_t prev_x = data->agent_x;
    size_t prev_y = data->agent_y;
    
    // Apply action
    switch (action_val) {
        case 0: // up
            if (data->agent_y > 0) data->agent_y--;
            break;
        case 1: // right
            if (data->agent_x < data->width - 1) data->agent_x++;
            break;
        case 2: // down
            if (data->agent_y < data->height - 1) data->agent_y++;
            break;
        case 3: // left
            if (data->agent_x > 0) data->agent_x--;
            break;
    }
    
    // Calculate reward
    float reward = data->reward_step;
    
    // Check if agent hit a wall (didn't move)
    if (data->agent_x == prev_x && data->agent_y == prev_y) {
        reward = data->reward_wall;
    }
    
    // Check if agent reached goal
    if (data->agent_x == data->goal_x && data->agent_y == data->goal_y) {
        reward = data->reward_goal;
        data->goal_reached = true;
        env->is_done = true;
    }
    
    // Check if max steps reached
    env->step_count++;
    if (env->step_count >= data->max_steps) {
        env->is_done = true;
    }
    
    // Create new observation
    cnp_tensor_t *next_obs = cnp_rl_gridworld_create_obs(data);
    
    // Update environment's current observation
    if (env->current_obs) {
        cnp_tensor_decref(env->current_obs);
    }
    env->current_obs = next_obs;
    cnp_tensor_incref(next_obs);
    
    // Create reward tensor
    size_t reward_dims[] = {1, 1};
    cnp_shape_t reward_shape = cnp_shape_create(2, reward_dims);
    cnp_tensor_t *reward_tensor = cnp_tensor_create(&reward_shape, CNP_FLOAT32, &reward);
    
    // Create step
    cnp_rl_step_t *step = cnp_rl_step_create(prev_obs, next_obs, action, reward_tensor, env->is_done, NULL);
    
    cnp_shape_free(&reward_shape);
    cnp_tensor_decref(prev_obs);
    
    return step;
}

static void cnp_rl_gridworld_render(cnp_rl_env_t *env) {
    cnp_rl_gridworld_data_t *data = (cnp_rl_gridworld_data_t*)env->env_data;
    
    printf("\nGrid World (%zu x %zu):\n", data->width, data->height);
    printf("Agent at (%zu, %zu), Goal at (%zu, %zu)\n", 
           data->agent_x, data->agent_y, data->goal_x, data->goal_y);
    printf("Steps: %zu/%zu\n", env->step_count, data->max_steps);
    
    for (size_t y = 0; y < data->height; y++) {
        for (size_t x = 0; x < data->width; x++) {
            if (x == data->agent_x && y == data->agent_y) {
                printf("A ");
            } else if (x == data->goal_x && y == data->goal_y) {
                printf("G ");
            } else {
                printf(". ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

static void cnp_rl_gridworld_cleanup(cnp_rl_env_t *env) {
    if (env->env_data) {
        free(env->env_data);
        env->env_data = NULL;
    }
}

cnp_rl_env_t* cnp_rl_gridworld_create(size_t width, size_t height) {
    // Create observation shape (flattened grid)
    size_t obs_dims[] = {1, width * height};
    cnp_shape_t obs_shape = cnp_shape_create(2, obs_dims);
    
    // Create action shape (single discrete action)
    size_t action_dims[] = {1, 1};
    cnp_shape_t action_shape = cnp_shape_create(2, action_dims);
    
    // Create environment
    cnp_rl_env_t *env = cnp_rl_env_create("GridWorld", &obs_shape, &action_shape, 4);
    
    // Create grid world specific data
    cnp_rl_gridworld_data_t *data = malloc(sizeof(cnp_rl_gridworld_data_t));
    data->width = width;
    data->height = height;
    data->agent_x = 0;
    data->agent_y = 0;
    data->goal_x = width - 1;
    data->goal_y = height - 1;
    data->reward_goal = 10.0f;
    data->reward_step = -0.1f;
    data->reward_wall = -1.0f;
    data->max_steps = width * height * 2;  // Reasonable max steps
    data->goal_reached = false;
    
    // Set function pointers
    env->reset = cnp_rl_gridworld_reset;
    env->step = cnp_rl_gridworld_step;
    env->render = cnp_rl_gridworld_render;
    env->cleanup = cnp_rl_gridworld_cleanup;
    
    // Set environment data
    env->env_data = data;
    
    cnp_shape_free(&obs_shape);
    cnp_shape_free(&action_shape);
    
    return env;
}

// ============================================================================
// CartPole Environment Implementation
// ============================================================================

typedef struct {
    float cart_position;    // Position of cart on track
    float cart_velocity;    // Velocity of cart
    float pole_angle;       // Angle of pole from vertical (radians)
    float pole_velocity;    // Angular velocity of pole
    
    // Physics parameters
    float gravity;          // Gravitational acceleration
    float cart_mass;        // Mass of cart
    float pole_mass;        // Mass of pole
    float pole_length;      // Half-length of pole
    float force_magnitude;  // Force applied to cart
    float tau;              // Time step for Euler integration
    
    // Environment parameters
    float x_threshold;      // Cart position bounds
    float theta_threshold;  // Pole angle bounds (radians)
    size_t max_steps;       // Maximum episode length
    
    // Rendering parameters
    bool render_mode;       // Whether to render
    float reward_alive;     // Reward for staying alive
    float reward_done;      // Reward for terminal state
} cnp_rl_cartpole_data_t;

static cnp_rl_step_t* cnp_rl_cartpole_reset(cnp_rl_env_t *env) {
    cnp_rl_cartpole_data_t *data = (cnp_rl_cartpole_data_t*)env->env_data;
    
    // Reset state to small random values
    data->cart_position = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    data->cart_velocity = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    data->pole_angle = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    data->pole_velocity = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    env->step_count = 0;
    env->is_done = false;
    
    // Create observation tensor [cart_pos, cart_vel, pole_angle, pole_vel]
    float obs_data[4] = {
        data->cart_position,
        data->cart_velocity,
        data->pole_angle,
        data->pole_velocity
    };
    
    cnp_tensor_t *obs = cnp_tensor_create(&env->obs_shape, CNP_FLOAT32, obs_data);
    
    // Update current observation
    if (env->current_obs) {
        cnp_tensor_decref(env->current_obs);
    }
    env->current_obs = obs;
    cnp_tensor_incref(env->current_obs);
    
    // Create initial step
    cnp_rl_step_t *step = cnp_rl_step_create(obs, NULL, NULL, NULL, false, NULL);
    
    return step;
}

static cnp_rl_step_t* cnp_rl_cartpole_step(cnp_rl_env_t *env, cnp_tensor_t *action) {
    cnp_rl_cartpole_data_t *data = (cnp_rl_cartpole_data_t*)env->env_data;
    
    // Get action (0 = left, 1 = right)
    float action_value = cnp_tensor_get_float(action, (size_t[]){0});
    float force = (action_value > 0.5f) ? data->force_magnitude : -data->force_magnitude;
    
    // Store previous state
    cnp_tensor_t *prev_obs = env->current_obs;
    cnp_tensor_incref(prev_obs);
    
    // Physics simulation using Euler integration
    float x_dot = data->cart_velocity;
    float theta = data->pole_angle;
    float theta_dot = data->pole_velocity;
    
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    
    // Calculate accelerations
    float temp = (force + data->pole_mass * data->pole_length * theta_dot * theta_dot * sin_theta) / 
                 (data->cart_mass + data->pole_mass);
    float theta_acc = (data->gravity * sin_theta - cos_theta * temp) / 
                     (data->pole_length * (4.0f/3.0f - data->pole_mass * cos_theta * cos_theta / 
                                         (data->cart_mass + data->pole_mass)));
    float x_acc = temp - data->pole_mass * data->pole_length * theta_acc * cos_theta / 
                  (data->cart_mass + data->pole_mass);
    
    // Update state
    data->cart_position += data->tau * x_dot;
    data->cart_velocity += data->tau * x_acc;
    data->pole_angle += data->tau * theta_dot;
    data->pole_velocity += data->tau * theta_acc;
    
    // Check termination conditions
    bool done = false;
    if (fabsf(data->cart_position) > data->x_threshold ||
        fabsf(data->pole_angle) > data->theta_threshold ||
        env->step_count >= data->max_steps) {
        done = true;
    }
    
    // Calculate reward
    float reward = done ? data->reward_done : data->reward_alive;
    
    // Create new observation
    float obs_data[4] = {
        data->cart_position,
        data->cart_velocity,
        data->pole_angle,
        data->pole_velocity
    };
    
    cnp_tensor_t *obs = cnp_tensor_create(&env->obs_shape, CNP_FLOAT32, obs_data);
    
    // Update environment state
    if (env->current_obs) {
        cnp_tensor_decref(env->current_obs);
    }
    env->current_obs = obs;
    cnp_tensor_incref(env->current_obs);
    
    env->step_count++;
    env->is_done = done;
    
    // Create reward tensor
    cnp_tensor_t *reward_tensor = cnp_tensor_create(
        &(cnp_shape_t){.ndim = 1, .dims = (size_t[]){1}, .size = 1},
        CNP_FLOAT32, &reward
    );
    
    // Create step
    cnp_rl_step_t *step = cnp_rl_step_create(prev_obs, obs, action, reward_tensor, done, NULL);
    
    // Cleanup
    cnp_tensor_decref(prev_obs);
    cnp_tensor_decref(reward_tensor);
    
    return step;
}

static void cnp_rl_cartpole_render(cnp_rl_env_t *env) {
    cnp_rl_cartpole_data_t *data = (cnp_rl_cartpole_data_t*)env->env_data;
    
    if (!data->render_mode) return;
    
    // ASCII art rendering
    printf("\n");
    printf("CartPole Environment (Step %zu)\n", env->step_count);
    printf("Cart Position: %.3f, Cart Velocity: %.3f\n", data->cart_position, data->cart_velocity);
    printf("Pole Angle: %.3f, Pole Velocity: %.3f\n", data->pole_angle, data->pole_velocity);
    
    // Simple visualization
    int track_width = 41;
    int cart_pos = (int)((data->cart_position + data->x_threshold) / (2 * data->x_threshold) * (track_width - 1));
    cart_pos = cart_pos < 0 ? 0 : (cart_pos >= track_width ? track_width - 1 : cart_pos);
    
    // Draw track
    printf("Track: ");
    for (int i = 0; i < track_width; i++) {
        if (i == cart_pos) {
            printf("C");  // Cart
        } else {
            printf("-");
        }
    }
    printf("\n");
    
    // Draw pole (simplified)
    printf("Pole:  ");
    for (int i = 0; i < track_width; i++) {
        if (i == cart_pos) {
            if (fabsf(data->pole_angle) < 0.1f) {
                printf("|");  // Upright pole
            } else if (data->pole_angle > 0) {
                printf("/");  // Leaning right
            } else {
                printf("\\"); // Leaning left
            }
        } else {
            printf(" ");
        }
    }
    printf("\n");
    
    printf("Reward: %.1f, Done: %s\n", 
           env->is_done ? data->reward_done : data->reward_alive,
           env->is_done ? "Yes" : "No");
    printf("----------------------------------------\n");
}

static void cnp_rl_cartpole_cleanup(cnp_rl_env_t *env) {
    if (env->env_data) {
        free(env->env_data);
        env->env_data = NULL;
    }
}

cnp_rl_env_t* cnp_rl_cartpole_create(void) {
    // Create environment
    size_t obs_dims[] = {4};
    size_t action_dims[] = {1};
    cnp_shape_t obs_shape = cnp_shape_create(1, obs_dims);
    cnp_shape_t action_shape = cnp_shape_create(1, action_dims);
    cnp_rl_env_t *env = cnp_rl_env_create("CartPole-v1", &obs_shape, &action_shape, 2);
    
    // Create CartPole-specific data
    cnp_rl_cartpole_data_t *data = malloc(sizeof(cnp_rl_cartpole_data_t));
    
    // Set default parameters (based on OpenAI Gym CartPole-v1)
    data->gravity = 9.8f;
    data->cart_mass = 1.0f;
    data->pole_mass = 0.1f;
    data->pole_length = 0.5f;  // Half-length
    data->force_magnitude = 10.0f;
    data->tau = 0.02f;  // 50 FPS
    data->x_threshold = 2.4f;
    data->theta_threshold = 12.0f * M_PI / 180.0f;  // 12 degrees in radians
    data->max_steps = 500;
    data->render_mode = false;
    data->reward_alive = 1.0f;
    data->reward_done = 0.0f;
    
    // Initialize state
    data->cart_position = 0.0f;
    data->cart_velocity = 0.0f;
    data->pole_angle = 0.0f;
    data->pole_velocity = 0.0f;
    
    // Set environment data and functions
    env->env_data = data;
    env->reset = cnp_rl_cartpole_reset;
    env->step = cnp_rl_cartpole_step;
    env->render = cnp_rl_cartpole_render;
    env->cleanup = cnp_rl_cartpole_cleanup;
    
    return env;
}

// ============================================================================
// MountainCar Environment Implementation
// ============================================================================

typedef struct {
    float position;         // Car position on hill
    float velocity;         // Car velocity
    
    // Environment parameters
    float min_position;     // Minimum position
    float max_position;     // Maximum position
    float goal_position;    // Goal position
    float min_velocity;     // Minimum velocity
    float max_velocity;     // Maximum velocity
    float force;            // Force applied by actions
    float gravity;          // Gravitational constant
    size_t max_steps;       // Maximum episode length
    
    // Rendering parameters
    bool render_mode;       // Whether to render
    float reward_step;      // Reward per step
    float reward_goal;      // Reward for reaching goal
} cnp_rl_mountaincar_data_t;

static cnp_rl_step_t* cnp_rl_mountaincar_reset(cnp_rl_env_t *env) {
    cnp_rl_mountaincar_data_t *data = (cnp_rl_mountaincar_data_t*)env->env_data;
    
    // Reset to random position with zero velocity
    data->position = ((float)rand() / RAND_MAX) * 0.2f - 0.6f;  // [-0.6, -0.4]
    data->velocity = 0.0f;
    
    env->step_count = 0;
    env->is_done = false;
    
    // Create observation tensor [position, velocity]
    float obs_data[2] = {data->position, data->velocity};
    cnp_tensor_t *obs = cnp_tensor_create(&env->obs_shape, CNP_FLOAT32, obs_data);
    
    // Update current observation
    if (env->current_obs) {
        cnp_tensor_decref(env->current_obs);
    }
    env->current_obs = obs;
    cnp_tensor_incref(env->current_obs);
    
    // Create initial step
    cnp_rl_step_t *step = cnp_rl_step_create(obs, NULL, NULL, NULL, false, NULL);
    
    return step;
}

static cnp_rl_step_t* cnp_rl_mountaincar_step(cnp_rl_env_t *env, cnp_tensor_t *action) {
    cnp_rl_mountaincar_data_t *data = (cnp_rl_mountaincar_data_t*)env->env_data;
    
    // Get action (0 = left, 1 = nothing, 2 = right)
    float action_value = cnp_tensor_get_float(action, (size_t[]){0});
    float force = (action_value - 1.0f) * data->force;  // Convert to [-1, 0, 1] * force
    
    // Store previous state
    cnp_tensor_t *prev_obs = env->current_obs;
    cnp_tensor_incref(prev_obs);
    
    // Physics simulation
    float velocity = data->velocity + force - data->gravity * cosf(3.0f * data->position);
    
    // Clamp velocity
    velocity = fmaxf(fminf(velocity, data->max_velocity), data->min_velocity);
    
    // Update position
    float position = data->position + velocity;
    
    // Clamp position and handle left boundary
    if (position < data->min_position) {
        position = data->min_position;
        velocity = 0.0f;  // Stop at left boundary
    } else if (position > data->max_position) {
        position = data->max_position;
    }
    
    // Update state
    data->position = position;
    data->velocity = velocity;
    
    // Check termination conditions
    bool done = false;
    float reward = data->reward_step;
    
    if (position >= data->goal_position) {
        done = true;
        reward = data->reward_goal;
    } else if (env->step_count >= data->max_steps) {
        done = true;
    }
    
    // Create new observation
    float obs_data[2] = {data->position, data->velocity};
    cnp_tensor_t *obs = cnp_tensor_create(&env->obs_shape, CNP_FLOAT32, obs_data);
    
    // Update environment state
    if (env->current_obs) {
        cnp_tensor_decref(env->current_obs);
    }
    env->current_obs = obs;
    cnp_tensor_incref(env->current_obs);
    
    env->step_count++;
    env->is_done = done;
    
    // Create reward tensor
    cnp_tensor_t *reward_tensor = cnp_tensor_create(
        &(cnp_shape_t){.ndim = 1, .dims = (size_t[]){1}, .size = 1},
        CNP_FLOAT32, &reward
    );
    
    // Create step
    cnp_rl_step_t *step = cnp_rl_step_create(prev_obs, obs, action, reward_tensor, done, NULL);
    
    // Cleanup
    cnp_tensor_decref(prev_obs);
    cnp_tensor_decref(reward_tensor);
    
    return step;
}

static void cnp_rl_mountaincar_render(cnp_rl_env_t *env) {
    cnp_rl_mountaincar_data_t *data = (cnp_rl_mountaincar_data_t*)env->env_data;
    
    if (!data->render_mode) return;
    
    // ASCII art rendering
    printf("\n");
    printf("MountainCar Environment (Step %zu)\n", env->step_count);
    printf("Position: %.3f, Velocity: %.3f\n", data->position, data->velocity);
    
    // Simple visualization
    int track_width = 41;
    int car_pos = (int)((data->position - data->min_position) / 
                       (data->max_position - data->min_position) * (track_width - 1));
    car_pos = car_pos < 0 ? 0 : (car_pos >= track_width ? track_width - 1 : car_pos);
    
    // Draw mountain track (simplified)
    printf("Track: ");
    for (int i = 0; i < track_width; i++) {
        if (i == car_pos) {
            printf("C");  // Car
        } else if (i > track_width * 0.8f) {
            printf("^");  // Goal area
        } else {
            printf("~");  // Mountain
        }
    }
    printf("\n");
    
    printf("Goal: %.3f, Reward: %.1f, Done: %s\n", 
           data->goal_position, 
           data->position >= data->goal_position ? data->reward_goal : data->reward_step,
           env->is_done ? "Yes" : "No");
    printf("----------------------------------------\n");
}

static void cnp_rl_mountaincar_cleanup(cnp_rl_env_t *env) {
    if (env->env_data) {
        free(env->env_data);
        env->env_data = NULL;
    }
}

cnp_rl_env_t* cnp_rl_mountaincar_create(void) {
    // Create environment
    size_t obs_dims[] = {2};
    size_t action_dims[] = {1};
    cnp_shape_t obs_shape = cnp_shape_create(1, obs_dims);
    cnp_shape_t action_shape = cnp_shape_create(1, action_dims);
    cnp_rl_env_t *env = cnp_rl_env_create("MountainCar-v0", &obs_shape, &action_shape, 3);
    
    // Create MountainCar-specific data
    cnp_rl_mountaincar_data_t *data = malloc(sizeof(cnp_rl_mountaincar_data_t));
    
    // Set default parameters (based on OpenAI Gym MountainCar-v0)
    data->min_position = -1.2f;
    data->max_position = 0.6f;
    data->goal_position = 0.5f;
    data->min_velocity = -0.07f;
    data->max_velocity = 0.07f;
    data->force = 0.001f;
    data->gravity = 0.0025f;
    data->max_steps = 200;
    data->render_mode = false;
    data->reward_step = -1.0f;
    data->reward_goal = 0.0f;
    
    // Initialize state
    data->position = -0.5f;
    data->velocity = 0.0f;
    
    // Set environment data and functions
    env->env_data = data;
    env->reset = cnp_rl_mountaincar_reset;
    env->step = cnp_rl_mountaincar_step;
    env->render = cnp_rl_mountaincar_render;
    env->cleanup = cnp_rl_mountaincar_cleanup;
    
    return env;
}

// ============================================================================
// Training Infrastructure
// ============================================================================

cnp_rl_traj_t* cnp_rl_run_episode(cnp_rl_agent_t *agent, cnp_rl_env_t *env,
                                  size_t max_steps, bool render) {
    if (!agent || !env) return NULL;
    
    // Reset environment
    cnp_rl_step_t *initial_step = env->reset(env);
    if (!initial_step) return NULL;
    
    // Create trajectory
    cnp_rl_traj_t *traj = cnp_rl_traj_create(1);
    
    cnp_rl_step_t *current_step = initial_step;
    cnp_rl_step_incref(current_step);  // We own a reference to current_step
    size_t steps = 0;
    
    if (render) {
        env->render(env);
    }
    
    while (!env->is_done && steps < max_steps) {
        // Agent chooses action
        cnp_tensor_t *action = agent->forward(agent, current_step);
        
        if (!action) {
            printf("Error: Agent forward failed\n");
            break;
        }
        
        // Environment steps
        cnp_rl_step_t *next_step = env->step(env, action);
        
        if (!next_step) {
            printf("Error: Environment step failed\n");
            cnp_tensor_decref(action);
            break;
        }
        
        // Add step to trajectory (this increments next_step's ref count)
        cnp_rl_traj_add_step(traj, next_step);
        
        // Update current step
        cnp_rl_step_decref(current_step);
        current_step = next_step;
        cnp_rl_step_incref(current_step);  // We now own a reference to the new current_step
        
        steps++;
        
        if (render) {
            env->render(env);
        }
    }
    
    // Cleanup current step
    if (current_step) {
        cnp_rl_step_decref(current_step);
    }
    
    return traj;
}

cnp_rl_training_stats_t* cnp_rl_train_agent(cnp_rl_training_config_t *config) {
    if (!config || !config->agent || !config->env) return NULL;
    
    cnp_rl_training_stats_t *stats = malloc(sizeof(cnp_rl_training_stats_t));
    stats->total_reward = 0.0f;
    stats->average_reward = 0.0f;
    stats->episode_length = 0;
    stats->total_steps = 0;
    stats->loss = 0.0f;
    
    float episode_rewards[config->max_episodes];
    size_t episode_lengths[config->max_episodes];
    
    for (size_t episode = 0; episode < config->max_episodes; episode++) {
        // Run episode
        cnp_rl_traj_t *traj = cnp_rl_run_episode(config->agent, config->env,
                                                config->max_steps_per_episode, config->render);
        
        if (!traj) continue;
        
        // Calculate episode reward
        float episode_reward = 0.0f;
        for (size_t i = 0; i < traj->num_steps; i++) {
            if (traj->steps[i]->reward) {
                float *reward_data = (float*)traj->steps[i]->reward->data;
                episode_reward += reward_data[0];
            }
        }
        
        episode_rewards[episode] = episode_reward;
        episode_lengths[episode] = traj->num_steps;
        stats->total_steps += traj->num_steps;
        
        // Add to replay buffer if available
        if (config->replay_buffer) {
            for (size_t i = 0; i < traj->num_steps; i++) {
                cnp_rl_replay_buffer_add(config->replay_buffer, traj->steps[i]);
            }
        }
        
        // Train agent
        if (config->train_freq > 0 && episode % config->train_freq == 0) {
            if (config->replay_buffer && config->replay_buffer->size > 0) {
                cnp_rl_traj_t *train_traj = cnp_rl_replay_buffer_sample(config->replay_buffer);
                if (train_traj) {
                    config->agent->train(config->agent, train_traj);
                    cnp_rl_traj_decref(train_traj);
                }
            } else {
                config->agent->train(config->agent, traj);
            }
        }
        
        // Print progress
        if (config->verbose && episode % 10 == 0) {
            printf("Episode %zu: Reward=%.2f, Length=%zu\n", 
                   episode, episode_reward, traj->num_steps);
        }
        
        cnp_rl_traj_decref(traj);
    }
    
    // Calculate final statistics
    stats->total_reward = 0.0f;
    stats->episode_length = 0;
    
    for (size_t i = 0; i < config->max_episodes; i++) {
        stats->total_reward += episode_rewards[i];
        stats->episode_length += episode_lengths[i];
    }
    
    stats->average_reward = stats->total_reward / config->max_episodes;
    stats->episode_length = stats->episode_length / config->max_episodes;
    
    return stats;
} 