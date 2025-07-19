# Contributing to CNmpy RL

We welcome contributions to CNmpy RL! This guide will help you get started.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Community](#community)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Types of Contributions

We welcome many types of contributions:

- **Bug fixes** - Help us fix issues and improve stability
- **New features** - Implement new RL algorithms, environments, or utilities
- **Performance improvements** - Optimize existing code for speed or memory
- **Documentation** - Improve docs, add examples, write tutorials
- **Testing** - Add or improve test coverage
- **Code review** - Review pull requests from other contributors

### Before You Start

1. **Check existing issues** - Look for existing issues or discussions
2. **Create an issue** - For new features, create an issue to discuss first
3. **Fork the repository** - Create your own fork to work on
4. **Small changes first** - Start with small, focused changes

## 💻 Development Setup

### Prerequisites

- **C Compiler**: GCC 7+ or Clang 6+
- **CMake**: 3.12+
- **Git**: Latest version
- **Python**: 3.7+ (for development tools)

### Setup Instructions

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/cnumpy-rl.git
cd cnumpy-rl

# Add the upstream repository
git remote add upstream https://github.com/username/cnumpy-rl.git

# Install development dependencies
./scripts/setup_dev.sh

# Build in debug mode
mkdir build-debug && cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON -DENABLE_COVERAGE=ON ..
make -j$(nproc)

# Run tests to ensure everything works
make test
```

### Development Tools

We use several tools to maintain code quality:

- **clang-format**: Code formatting
- **clang-tidy**: Static analysis
- **valgrind**: Memory leak detection
- **gcov/lcov**: Code coverage analysis
- **pre-commit**: Git hooks for quality checks

Install pre-commit hooks:
```bash
pre-commit install
```

## 🔧 Making Changes

### Branch Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for new features
- `feature/feature-name`: New feature development
- `bugfix/issue-number`: Bug fixes
- `hotfix/issue-number`: Critical bug fixes

### Workflow

1. **Create a branch** from `develop`:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** following our code style
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Commit your changes** with clear messages
6. **Push to your fork** and create a pull request

### Commit Messages

Follow the conventional commits format:

```
type(scope): brief description

Longer description if needed.

Fixes #issue-number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements

Examples:
```
feat(agents): add PPO algorithm implementation

Add Proximal Policy Optimization algorithm with:
- Actor-critic architecture
- Clipped objective function
- Generalized advantage estimation
- Entropy regularization

Fixes #123
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
├── performance/    # Performance benchmarks
├── memory/         # Memory leak tests
└── examples/       # Example validation tests
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-performance
make test-memory

# Run tests with coverage
make coverage

# Run specific test
./build-debug/tests/unit/test_dqn_agent
```

### Writing Tests

All new features should include comprehensive tests:

1. **Unit tests** for individual functions
2. **Integration tests** for component interactions
3. **Performance tests** for critical paths
4. **Memory tests** for proper cleanup

Example test structure:
```c
// tests/unit/test_dqn_agent.c
#include <cnumpy_rl.h>
#include <assert.h>
#include <stdio.h>

void test_dqn_agent_creation() {
    cnp_rl_init();
    
    cnp_rl_agent_t *agent = cnp_rl_dqn_agent_create(
        "TestAgent", 4, 64, 2, 0.001f, 1.0f, 0.01f, 0.995f, 0.99f
    );
    
    assert(agent != NULL);
    assert(strcmp(agent->name, "TestAgent") == 0);
    assert(agent->ref_count == 1);
    
    cnp_rl_agent_decref(agent);
    cnp_rl_cleanup();
    
    printf("✓ DQN agent creation test passed\n");
}

int main() {
    test_dqn_agent_creation();
    return 0;
}
```

## 📚 Documentation

### Documentation Types

- **API Documentation**: Inline comments in header files
- **User Guide**: High-level usage documentation
- **Examples**: Complete working examples
- **Tutorials**: Step-by-step guides

### Documentation Style

- Use clear, concise language
- Include code examples
- Explain parameters and return values
- Document edge cases and limitations

Example documentation:
```c
/**
 * @brief Create a DQN agent for reinforcement learning
 * 
 * Creates a Deep Q-Network agent with the specified configuration.
 * The agent uses experience replay and target network for stable training.
 * 
 * @param name Human-readable name for the agent
 * @param obs_size Size of observation space
 * @param hidden_size Size of hidden layers in neural network
 * @param num_actions Number of discrete actions
 * @param learning_rate Learning rate for neural network optimization
 * @param epsilon_start Initial exploration rate
 * @param epsilon_end Final exploration rate
 * @param epsilon_decay Decay rate for exploration
 * @param gamma Discount factor for future rewards
 * 
 * @return Newly created DQN agent, or NULL on failure
 * 
 * @note The returned agent has reference count 1 and must be freed with
 *       cnp_rl_agent_decref() when no longer needed.
 * 
 * @example
 * ```c
 * cnp_rl_agent_t *agent = cnp_rl_dqn_agent_create(
 *     "MyAgent", 4, 128, 2, 0.001f, 1.0f, 0.01f, 0.995f, 0.99f
 * );
 * // Use agent...
 * cnp_rl_agent_decref(agent);
 * ```
 */
cnp_rl_agent_t* cnp_rl_dqn_agent_create(const char *name, size_t obs_size,
                                         size_t hidden_size, size_t num_actions,
                                         float learning_rate, float epsilon_start,
                                         float epsilon_end, float epsilon_decay,
                                         float gamma);
```

## 🔍 Pull Request Process

### Before Submitting

1. **Run all tests** and ensure they pass
2. **Check code coverage** - new code should have >90% coverage
3. **Run static analysis** - fix any warnings
4. **Update documentation** - including API docs and examples
5. **Test performance** - ensure no regressions

### PR Template

When creating a pull request, include:

- **Description**: Clear explanation of changes
- **Type**: Feature, bug fix, documentation, etc.
- **Testing**: How you tested the changes
- **Breaking changes**: Any API changes
- **Checklist**: Verification that requirements are met

### Review Process

1. **Automated checks** run on all PRs
2. **Code review** by maintainers
3. **Testing** on multiple platforms
4. **Documentation review** if applicable
5. **Merge** once approved

### Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- API design consistency
- Memory safety

## 🎨 Code Style

### C Style Guide

We follow a modified version of the Google C++ Style Guide:

#### Naming Conventions

- **Functions**: `snake_case` with `cnp_rl_` prefix
- **Types**: `snake_case` with `_t` suffix
- **Variables**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Macros**: `UPPER_CASE`

#### Code Formatting

- **Indentation**: 4 spaces, no tabs
- **Line length**: 100 characters maximum
- **Braces**: Opening brace on same line
- **Spacing**: Space after keywords, around operators

Example:
```c
// Good
cnp_rl_agent_t* cnp_rl_dqn_agent_create(const char *name, size_t obs_size) {
    if (!name || obs_size == 0) {
        return NULL;
    }
    
    cnp_rl_agent_t *agent = malloc(sizeof(cnp_rl_agent_t));
    if (!agent) {
        return NULL;
    }
    
    // Initialize agent...
    return agent;
}

// Bad
cnp_rl_agent_t* cnp_rl_dqn_agent_create(const char *name,size_t obs_size){
if(!name||obs_size==0){
return NULL;
}
cnp_rl_agent_t *agent=malloc(sizeof(cnp_rl_agent_t));
if(!agent){
return NULL;
}
return agent;
}
```

#### Memory Management

- Always use reference counting for objects
- Check for NULL pointers
- Free all allocated memory
- Use const correctness

#### Error Handling

- Return NULL for allocation failures
- Use assert() for programming errors
- Provide meaningful error messages
- Document error conditions

### Formatting Tools

Use clang-format with our configuration:
```bash
# Format all source files
find . -name "*.c" -o -name "*.h" | xargs clang-format -i

# Check formatting
make format-check
```

## 👥 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat with community
- **Email**: For security issues or private matters

### Getting Help

- Check the [documentation](https://cnumpy-rl.readthedocs.io)
- Search existing issues and discussions
- Ask questions in GitHub Discussions
- Join our Discord community

### Reporting Issues

When reporting bugs, include:

- **Environment**: OS, compiler, version
- **Steps to reproduce**: Minimal example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error output

## 🏆 Recognition

Contributors are recognized in several ways:

- **Contributors list** in README
- **Changelog** mentions for significant contributions
- **Release notes** for major features
- **Discord roles** for active contributors

## 📄 License

By contributing to CNmpy RL, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

---

Thank you for contributing to CNmpy RL! Your efforts help make reinforcement learning more accessible and efficient for everyone. 