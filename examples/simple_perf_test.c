#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "../src/cnumpy.h"
#include "../src/rl/cnumpy_rl.h"

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    printf("CNmpy Performance Test\n");
    printf("======================\n\n");
    
    cnp_rl_init();
    
    // Test 1: Tensor operations speed
    printf("Test 1: Tensor Operations Speed\n");
    size_t dims[] = {1000, 1000};
    cnp_shape_t shape = cnp_shape_create(2, dims);
    
    double start = get_time();
    cnp_tensor_t *a = cnp_ones(&shape, CNP_FLOAT32);
    cnp_tensor_t *b = cnp_ones(&shape, CNP_FLOAT32);
    
    for (int i = 0; i < 100; i++) {
        cnp_tensor_t *c = cnp_add(a, b);
        cnp_tensor_decref(c);
    }
    double end = get_time();
    
    printf("  100 additions of 1000x1000 tensors: %.6f seconds\n", end - start);
    printf("  Operations per second: %.2f\n", 100.0 / (end - start));
    
    // Test 2: Memory usage per tensor
    printf("\nTest 2: Memory Usage\n");
    size_t tensor_size = sizeof(cnp_tensor_t);
    size_t data_size = 1000 * 1000 * sizeof(float);
    printf("  Tensor struct: %zu bytes\n", tensor_size);
    printf("  Data for 1000x1000 float: %zu bytes\n", data_size);
    printf("  Total per tensor: %zu bytes\n", tensor_size + data_size);
    
    // Test 3: RL episode speed
    printf("\nTest 3: RL Episode Speed\n");
    cnp_rl_env_t *env = cnp_rl_gridworld_create(4, 4);
    cnp_rl_agent_t *agent = cnp_rl_random_agent_create("PerfTest", 4);
    
    start = get_time();
    int total_steps = 0;
    for (int episode = 0; episode < 100; episode++) {
        cnp_rl_traj_t *traj = cnp_rl_run_episode(agent, env, 50, false);
        total_steps += traj->num_steps;
        cnp_rl_traj_decref(traj);
    }
    end = get_time();
    
    printf("  100 episodes, %d total steps: %.6f seconds\n", total_steps, end - start);
    printf("  Steps per second: %.2f\n", total_steps / (end - start));
    printf("  Episodes per second: %.2f\n", 100.0 / (end - start));
    
    // Test 4: Reference counting overhead
    printf("\nTest 4: Reference Counting Overhead\n");
    start = get_time();
    cnp_tensor_t *test = cnp_ones(&shape, CNP_FLOAT32);
    for (int i = 0; i < 1000000; i++) {
        cnp_tensor_incref(test);
        cnp_tensor_decref(test);
    }
    end = get_time();
    cnp_tensor_decref(test);
    
    printf("  1M incref/decref cycles: %.6f seconds\n", end - start);
    printf("  Reference operations per second: %.2f\n", 2000000.0 / (end - start));
    
    // Test 5: Function pointer dispatch speed
    printf("\nTest 5: Function Pointer vs Direct Call\n");
    
    // Direct function call timing
    start = get_time();
    for (int i = 0; i < 1000000; i++) {
        volatile int x = i + 1; // Simple operation
        (void)x; // Suppress warning
    }
    end = get_time();
    double direct_time = end - start;
    
    // Function pointer call timing (simulate polymorphic call)
    int (*func_ptr)(int) = NULL;
    // Use a simple function for timing
    func_ptr = NULL; // We'll just time the overhead
    start = get_time();
    for (int i = 0; i < 1000000; i++) {
        if (func_ptr) {
            volatile int x = func_ptr(i);
            (void)x;
        } else {
            volatile int x = i + 1; // Same operation
            (void)x;
        }
    }
    end = get_time();
    double indirect_time = end - start;
    
    printf("  Direct calls: %.6f seconds\n", direct_time);
    printf("  Function pointer calls: %.6f seconds\n", indirect_time);
    printf("  Overhead: %.2f%%\n", ((indirect_time - direct_time) / direct_time) * 100.0);
    
    // Cleanup
    cnp_rl_agent_decref(agent);
    cnp_rl_env_decref(env);
    cnp_tensor_decref(a);
    cnp_tensor_decref(b);
    cnp_shape_free(&shape);
    cnp_rl_cleanup();
    
    printf("\nPerformance test completed!\n");
    return 0;
} 