#include "../headers/tasks.hpp"

#define FP double
static uint8_t solver_num;
static uint32_t n, m, max_iterations;
static FP eps, omega;

extern "C" void solve_test_task(uint8_t solver_num, uint32_t n, uint32_t m, uint32_t max_iterations, FP eps, FP omega) {
    test_task(solver_num, n, m, max_iterations, eps, omega);
}


extern "C" void solve_test_custom_task(uint8_t solver_num, uint32_t n, uint32_t m, uint32_t max_iterations, FP eps, FP omega) {
    test_custom_task(solver_num, n, m, max_iterations, eps, omega);
}


extern "C" void solve_main_task(uint8_t solver_num, uint32_t n, uint32_t m, uint32_t max_iterations, FP eps, FP omega) {
    main_task(solver_num, n, m, max_iterations, eps, omega);
}