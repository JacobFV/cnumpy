# CNmpy RL Architecture Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Design Principles](#core-design-principles)
3. [Architecture Components](#architecture-components)
4. [Data Flow](#data-flow)
5. [Memory Management](#memory-management)
6. [API Design](#api-design)
7. [Performance Considerations](#performance-considerations)
8. [Extension Points](#extension-points)
9. [Implementation Details](#implementation-details)
10. [Best Practices](#best-practices)

## Overview

CNmpy RL is a high-performance reinforcement learning library implemented in C, designed for both research and production use. The architecture emphasizes modularity, performance, and ease of use while maintaining the flexibility needed for advanced RL research.

### Key Design Goals

- **Performance**: Minimize computational overhead and memory usage
- **Modularity**: Clean separation of concerns with well-defined interfaces
- **Extensibility**: Easy to add new algorithms, environments, and features
- **Safety**: Memory-safe with automatic resource management
- **Simplicity**: Intuitive API that doesn't sacrifice functionality

## Core Design Principles

### 1. Polymorphism Through Function Pointers

CNmpy RL uses function pointers to achieve polymorphism in C, allowing different implementations of environments, agents, and algorithms to share common interfaces.

```c
typedef struct cnp_rl_env {
    // Function pointers for polymorphic behavior
    cnp_rl_env_reset_fn_t reset;
    cnp_rl_env_step_fn_t step;
    cnp_rl_env_render_fn_t render;
    cnp_rl_env_cleanup_fn_t cleanup;
    
    // Environment-specific data
    void *env_data;
} cnp_rl_env_t;
```

### 2. Reference Counting Memory Management

All objects use reference counting for automatic memory management, preventing memory leaks and double-free errors.

```c
void cnp_rl_env_incref(cnp_rl_env_t *env) {
    if (env) {
        env->ref_count++;
    }
}

void cnp_rl_env_decref(cnp_rl_env_t *env) {
    if (env && --env->ref_count <= 0) {
        cnp_rl_env_free(env);
    }
}
```

### 3. Immutable Data Structures

Core data structures like steps and trajectories are designed to be immutable after creation, reducing bugs and enabling safe concurrent access.

### 4. Modular Architecture

The library is organized into logical modules:

- **Core**: Basic data structures and utilities
- **Environments**: RL environment implementations
- **Agents**: RL algorithm implementations
- **Training**: Training loops and utilities

## Architecture Components

### 1. Core Data Structures

#### Step (`cnp_rl_step_t`)
Represents a single environment transition:
```c
typedef struct cnp_rl_step {
    cnp_tensor_t *obs;          // Current observation
    cnp_tensor_t *next_obs;     // Next observation
    cnp_tensor_t *action;       // Action taken
    cnp_tensor_t *reward;       // Reward received
    bool done;                  // Episode terminal flag
    void *info;                 // Additional metadata
    size_t batch_size;          // Batch dimension size
    int ref_count;              // Reference counting
} cnp_rl_step_t;
```

#### Trajectory (`cnp_rl_traj_t`)
Represents a sequence of steps (an episode):
```c
typedef struct cnp_rl_traj {
    cnp_rl_step_t **steps;      // Array of steps
    size_t num_steps;           // Number of steps
    size_t capacity;            // Allocated capacity
    size_t batch_size;          // Batch size for all steps
    int ref_count;              // Reference counting
} cnp_rl_traj_t;
```

#### Environment (`cnp_rl_env_t`)
Abstract environment interface:
```c
typedef struct cnp_rl_env {
    char *name;                 // Environment name
    cnp_tensor_t *current_obs;  // Current observation
    cnp_shape_t obs_shape;      // Observation space shape
    cnp_shape_t action_shape;   // Action space shape
    size_t num_actions;         // Number of discrete actions
    
    // Polymorphic functions
    cnp_rl_env_reset_fn_t reset;
    cnp_rl_env_step_fn_t step;
    cnp_rl_env_render_fn_t render;
    cnp_rl_env_cleanup_fn_t cleanup;
    
    void *env_data;             // Environment-specific data
    int ref_count;              // Reference counting
} cnp_rl_env_t;
```

#### Agent (`cnp_rl_agent_t`)
Abstract agent interface:
```c
typedef struct cnp_rl_agent {
    char *name;                 // Agent name
    cnp_tensor_t **parameters;  // Trainable parameters
    size_t num_parameters;      // Number of parameters
    
    // Hyperparameters
    float learning_rate;
    float epsilon;
    float gamma;
    
    // Polymorphic functions
    cnp_rl_agent_forward_fn_t forward;
    cnp_rl_agent_reward_fn_t reward;
    cnp_rl_agent_train_fn_t train;
    cnp_rl_agent_cleanup_fn_t cleanup;
    
    void *agent_data;           // Agent-specific data
    int ref_count;              // Reference counting
} cnp_rl_agent_t;
```

### 2. Environment Implementations

#### GridWorld Environment
A simple grid-based navigation environment for testing:
- **State Space**: One-hot encoded grid positions
- **Action Space**: 4 discrete actions (up, right, down, left)
- **Reward Structure**: Goal (+10), wall collision (-1), step (-0.1)

#### CartPole Environment
Classic control benchmark with realistic physics:
- **State Space**: [cart_position, cart_velocity, pole_angle, pole_velocity]
- **Action Space**: 2 discrete actions (left, right)
- **Physics**: Euler integration with proper dynamics
- **Termination**: Cart position > 2.4 or pole angle > 12 degrees

#### MountainCar Environment
Sparse reward environment requiring momentum:
- **State Space**: [position, velocity]
- **Action Space**: 3 discrete actions (left, none, right)
- **Physics**: Sinusoidal hill with gravity
- **Goal**: Reach position 0.5 on the right side

### 3. Agent Implementations

#### Random Agent
Baseline agent for comparison:
- Uniform random action selection
- No learning or parameters
- Useful for environment validation

#### DQN Agent
Deep Q-Network implementation:
- **Neural Network**: 2-layer fully connected
- **Experience Replay**: Circular buffer with batch sampling
- **Target Network**: Separate target network for stability
- **Epsilon-Greedy**: Exploration with decay
- **Training**: Batch gradient descent with MSE loss

## Data Flow

### 1. Training Loop

```
Initialize → Reset Environment → Agent Action → Environment Step → Store Experience → Train Agent → Repeat
```

### 2. Detailed Flow

1. **Environment Reset**: `env->reset(env)` returns initial step
2. **Agent Action**: `agent->forward(agent, step)` returns action tensor
3. **Environment Step**: `env->step(env, action)` returns next step
4. **Experience Storage**: Step added to trajectory
5. **Agent Training**: `agent->train(agent, trajectory)` updates parameters
6. **Cleanup**: Reference counting manages memory

### 3. Memory Flow

```
Step Creation → Trajectory Addition → Agent Training → Gradient Computation → Parameter Update → Cleanup
```

## Memory Management

### 1. Reference Counting

Every object maintains a reference count:
- **Creation**: `ref_count = 1`
- **Increment**: `cnp_rl_*_incref(obj)`
- **Decrement**: `cnp_rl_*_decref(obj)`
- **Cleanup**: When `ref_count <= 0`, object is freed

### 2. Ownership Rules

- **Creator Owns**: The function that creates an object owns the initial reference
- **Borrower Increments**: Functions that store references must increment
- **Caller Decrements**: Callers must decrement when done
- **Automatic Cleanup**: Objects are freed when no references remain

### 3. Memory Patterns

```c
// Creation pattern
cnp_rl_env_t *env = cnp_rl_cartpole_create();  // ref_count = 1

// Usage pattern
cnp_rl_env_incref(env);  // ref_count = 2
some_function(env);      // Function may store reference
cnp_rl_env_decref(env);  // ref_count = 1

// Cleanup pattern
cnp_rl_env_decref(env);  // ref_count = 0, object freed
```

## API Design

### 1. Naming Conventions

- **Prefix**: All functions use `cnp_rl_` prefix
- **Module**: Module name follows prefix (e.g., `cnp_rl_env_`)
- **Action**: Action follows module (e.g., `cnp_rl_env_create`)

### 2. Parameter Ordering

1. **Target Object**: Object being operated on
2. **Required Parameters**: Essential parameters
3. **Optional Parameters**: Optional parameters with defaults
4. **Output Parameters**: Output parameters (if any)

### 3. Error Handling

- **Return Values**: `NULL` for allocation failures
- **Assertions**: `assert()` for programming errors
- **Validation**: Input validation with early returns

### 4. Const Correctness

- **Read-only**: Parameters that aren't modified are `const`
- **Immutable**: Data structures that shouldn't change are `const`
- **Thread Safety**: `const` helps with concurrent access

## Performance Considerations

### 1. Memory Allocation

- **Pooling**: Memory pools for frequently allocated objects
- **Batch Operations**: Vectorized operations where possible
- **Cache Locality**: Data structures designed for cache efficiency

### 2. Computational Efficiency

- **SIMD**: Vectorized operations for tensor computations
- **Inlining**: Critical functions marked `inline`
- **Branch Prediction**: Predictable branches in hot paths

### 3. Scalability

- **Parallel Training**: Support for multi-threaded training
- **Batch Processing**: Efficient batch operations
- **Memory Bounds**: Configurable memory limits

## Extension Points

### 1. New Environments

```c
// 1. Define environment-specific data
typedef struct {
    // Your environment state
} my_env_data_t;

// 2. Implement interface functions
static cnp_rl_step_t* my_env_reset(cnp_rl_env_t *env);
static cnp_rl_step_t* my_env_step(cnp_rl_env_t *env, cnp_tensor_t *action);
static void my_env_render(cnp_rl_env_t *env);
static void my_env_cleanup(cnp_rl_env_t *env);

// 3. Create constructor
cnp_rl_env_t* cnp_rl_my_env_create(void);
```

### 2. New Agents

```c
// 1. Define agent-specific data
typedef struct {
    // Your agent state
} my_agent_data_t;

// 2. Implement interface functions
static cnp_tensor_t* my_agent_forward(cnp_rl_agent_t *agent, cnp_rl_step_t *step);
static float my_agent_reward(cnp_rl_agent_t *agent, cnp_rl_traj_t *traj);
static void my_agent_train(cnp_rl_agent_t *agent, cnp_rl_traj_t *traj);
static void my_agent_cleanup(cnp_rl_agent_t *agent);

// 3. Create constructor
cnp_rl_agent_t* cnp_rl_my_agent_create(const char *name, /* parameters */);
```

### 3. New Algorithms

The agent interface supports arbitrary algorithms:
- **Value-based**: DQN, Double DQN, Dueling DQN
- **Policy-based**: REINFORCE, PPO, A3C
- **Actor-Critic**: A2C, SAC, TD3
- **Model-based**: MCTS, Dyna-Q

## Implementation Details

### 1. Tensor Integration

CNmpy RL builds on the core CNmpy tensor library:
- **Automatic Differentiation**: Gradients computed automatically
- **Memory Management**: Tensors use reference counting
- **Operations**: Rich set of tensor operations available

### 2. Neural Networks

Neural networks are implemented as collections of tensors:
```c
// Network structure
typedef struct {
    cnp_tensor_t *W1, *b1;  // First layer
    cnp_tensor_t *W2, *b2;  // Second layer
} network_t;

// Forward pass
cnp_tensor_t* network_forward(network_t *net, cnp_tensor_t *input) {
    cnp_tensor_t *h1 = cnp_relu(cnp_add(cnp_matmul(input, net->W1), net->b1));
    cnp_tensor_t *output = cnp_add(cnp_matmul(h1, net->W2), net->b2);
    return output;
}
```

### 3. Training Infrastructure

Training is implemented as a configurable pipeline:
```c
typedef struct {
    cnp_rl_agent_t *agent;
    cnp_rl_env_t *env;
    size_t max_episodes;
    size_t max_steps_per_episode;
    bool render;
    bool verbose;
} cnp_rl_training_config_t;
```

### 4. Optimization

The library includes several optimization techniques:
- **Memory Pooling**: Reuse objects to reduce allocations
- **Batch Processing**: Process multiple samples together
- **Vectorization**: Use SIMD for tensor operations
- **Caching**: Cache frequently computed values

## Best Practices

### 1. Memory Management

- **Always** call `decref` when done with objects
- **Never** use objects after calling `decref`
- **Check** return values for `NULL`
- **Use** RAII patterns where possible

### 2. Performance

- **Profile** before optimizing
- **Measure** memory usage and allocations
- **Batch** operations when possible
- **Reuse** objects to reduce allocations

### 3. Error Handling

- **Validate** inputs at API boundaries
- **Handle** allocation failures gracefully
- **Assert** programming invariants
- **Log** errors for debugging

### 4. Testing

- **Unit test** individual components
- **Integration test** complete workflows
- **Memory test** for leaks and corruption
- **Performance test** for regressions

### 5. Documentation

- **Document** all public APIs
- **Provide** examples for complex usage
- **Explain** design decisions
- **Update** documentation with changes

## Conclusion

CNmpy RL's architecture provides a solid foundation for high-performance reinforcement learning in C. The modular design, reference counting memory management, and polymorphic interfaces enable both performance and flexibility while maintaining code clarity and safety.

The architecture supports a wide range of RL algorithms and environments while providing the performance characteristics needed for production use. The extension points allow researchers and practitioners to add new algorithms and environments without modifying the core library.

This design has been validated through extensive testing and benchmarking, demonstrating both correctness and performance across a variety of RL tasks and environments. 