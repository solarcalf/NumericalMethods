#include <vector>
#include <algorithm>
#include <memory>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cmath>

#include "../headers/Dirichlet_Problem.hpp"
#include "../headers/Solvers.hpp"

#define FP double

template<numcpp::GridType Grid>
std::pair<size_t, std::vector<std::vector<FP>>> estimate_time(numcpp::DirichletProblemSolver<Grid>& solver)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto solution = solver.solve();
    auto end = std::chrono::high_resolution_clock::now();

    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Overall time: " << duration / 1000000 << "s\n";

    return {duration, solution};
}

void test_TopRelaxation(){
    size_t n = 2048;
    size_t m = 2048;
    std::cout << "n = " << n << " m = " << m << '\n';

    FP omega = 1.95;
    std::array<double, 4> corners = {-1.0, -1.0, 1.0, 1.0};
    std::vector<FP> init_app((n - 1) * (m - 1), 0.0);

    auto u = [](double x, double y) { return exp(1 - pow(x, 2) - pow(y, 2)); };
    auto f = [](double x, double y) { return -4 * exp(1 - pow(x, 2) - pow(y, 2)) * (pow(y, 2) + pow(x, 2) - 1); };

    auto mu1 = [](double y) { return exp(-pow(y, 2)); };
    auto mu2 = [](double y) { return exp(-pow(y, 2)); };
    auto mu3 = [](double x) { return exp(-pow(x, 2)); };
    auto mu4 = [](double x) { return exp(-pow(x, 2)); };

    numcpp::DirichletProblemSolver<numcpp::Regular> dirichlet_task;
    dirichlet_task.set_fraction(m, n);
    dirichlet_task.set_corners(corners);
    dirichlet_task.set_u(u);
    dirichlet_task.set_f(f);
    dirichlet_task.set_boundary_conditions({mu1, mu2, mu3, mu4});

    //dirichlet_task.set_solver(std::make_unique<numcpp::TopRelaxation>(init_app, 10000, 0.00001, nullptr, std::vector<FP>(), omega));
    dirichlet_task.set_solver(std::make_unique<numcpp::TopRelaxationOptimizedForDirichletRegularGrid>
    (init_app, 1000000000, 0.0000000000001, nullptr, std::vector<FP>(), f, mu1, mu2, mu3, mu4, n, m, corners, omega));

    auto [duration, solution] = estimate_time(dirichlet_task);
}

void test_task(size_t m, size_t n, std::unique_ptr<numcpp::ISolver> LS_solver)
{
    std::cout << "n = " << n << " m = " << m << '\n';
    size_t size = (n - 1) * (m - 1);
    
    std::array<double, 4> corners = {-1.0, -1.0, 1.0, 1.0};

    std::vector<FP> initial_approximation(size, 0.0);
    LS_solver->set_initial_approximation(initial_approximation);

    auto u = [](double x, double y) { return exp(1 - pow(x, 2) - pow(y, 2)); };
    auto f = [](double x, double y) { return -4 * exp(1 - pow(x, 2) - pow(y, 2)) * (pow(y, 2) + pow(x, 2) - 1); };
    auto mu1 = [](double y) { return exp(-pow(y, 2)); };
    auto mu2 = [](double y) { return exp(-pow(y, 2)); };
    auto mu3 = [](double x) { return exp(-pow(x, 2)); };
    auto mu4 = [](double x) { return exp(-pow(x, 2)); };

    numcpp::DirichletProblemSolver<numcpp::Regular> solver;

    solver.set_fraction(m, n);
    solver.set_corners(corners);
    solver.set_u(u);
    solver.set_f(f);
    solver.set_boundary_conditions({mu1, mu2, mu3, mu4});
    solver.set_solver(std::move(LS_solver));

    auto [duration, solution] = estimate_time(solver);
}

void test_task_custom_grid(size_t m, size_t n, std::unique_ptr<numcpp::ISolver> LS_solver)
{
    size_t size = (n / 2 - 1) * (m - 1) + (n / 2) * (m / 2 - 1);
    std::cout << "n = " << n << " m = " << m << '\n';

    std::vector<FP> initial_approximation(size, 0.0);
    LS_solver->set_initial_approximation(initial_approximation);

    std::array<double, 4> corners = {-1.0, -1.0, 1.0, 1.0};

    auto u = [](double x, double y) { return exp(1 - pow(x, 2) - pow(y, 2)); };
    auto f = [](double x, double y) { return -4 * exp(1 - pow(x, 2) - pow(y, 2)) * (pow(y, 2) + pow(x, 2) - 1); };

    auto mu1 = [](double y) { return exp(-pow(y, 2)); };
    auto mu2 = [](double y) { return exp(1.0 - pow(y, 2)); };
    auto mu3 = [](double y) { return exp(-pow(y, 2)); };
    auto mu4 = [](double x) { return exp(-pow(x, 2)); };
    auto mu5 = [](double x) { return exp(1.0 - pow(x, 2)); };
    auto mu6 = [](double x) { return exp(-pow(x, 2)); };

    numcpp::DirichletProblemSolver<numcpp::GridType::ReversedR> solver;
    std::array<std::function<FP(FP)>, 6> arr{mu1, mu2, mu3, mu4, mu5, mu6};

    solver.set_solver(std::move(LS_solver));
    solver.set_boundary_conditions_for_r_shaped_grid(arr);
    solver.set_fraction(n, m);
    solver.set_corners(corners);
    solver.set_u(u);
    solver.set_f(f);

    auto [duration, solution] = estimate_time(solver);
}

void test_Chebyshev(size_t n, size_t m) {
    // size_t n = 2048;
    // size_t m = 2048;
    std::cout << "n = " << n << " m = " << m << '\n';

    std::array<double, 4> corners = { -1.0, -1.0, 1.0, 1.0 };
    std::vector<FP> init_app((n - 1) * (m - 1), 0.0);

    FP h = (corners[2] - corners[0]) / n;
    FP k = (corners[3] - corners[1]) / m;

    FP Mmin = 4.0 / pow(h, 2) * pow(sin(numcpp::PI / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI / 2.0 / m), 2);
    FP Mmax = 4.0 / pow(h, 2) * pow(sin(numcpp::PI * (n - 1) / 2.0 / n), 2) + 4.0 / pow(k, 2) * pow(sin(numcpp::PI * (m - 1) / 2.0 / m), 2);
    std::cout << "Mmin = " << Mmin << " Mmax = " << Mmax << '\n';
    auto u = [](double x, double y) { return exp(1 - pow(x, 2) - pow(y, 2)); };
    auto f = [](double x, double y) { return -4 * exp(1 - pow(x, 2) - pow(y, 2)) * (pow(y, 2) + pow(x, 2) - 1); };

    auto mu1 = [](double y) { return exp(-pow(y, 2)); };
    auto mu2 = [](double y) { return exp(-pow(y, 2)); };
    auto mu3 = [](double x) { return exp(-pow(x, 2)); };
    auto mu4 = [](double x) { return exp(-pow(x, 2)); };

    numcpp::DirichletProblemSolver<numcpp::Regular> dirichlet_task;
    dirichlet_task.set_fraction(m, n);
    dirichlet_task.set_corners(corners);
    dirichlet_task.set_u(u);
    dirichlet_task.set_f(f);
    dirichlet_task.set_boundary_conditions({ mu1, mu2, mu3, mu4 });

    dirichlet_task.set_solver(std::make_unique<numcpp::ChebyshevIteration>(init_app, 1000000000, 0.0000000000001, nullptr, std::vector<FP>(), Mmin, Mmax));

    auto [duration, solution] = estimate_time(dirichlet_task);
}

int main() 
{
    size_t m = 200;
    size_t n = 200;

    std::cout << "Conjugate gradient method" << std::endl;
    auto LS_solver1 = std::make_unique<numcpp::ConGrad>(std::vector<FP>(), 1000000, 0.000000001, nullptr, std::vector<FP>());
    test_task_custom_grid(m, n, std::move(LS_solver1));

    std::cout << "\nMinimal residual method" << std::endl;
    auto LS_solver2 = std::make_unique<numcpp::MinRes>(std::vector<FP>(), 1000000, 0.000001, nullptr, std::vector<FP>());
    test_task(m, n, std::move(LS_solver2));

    std::cout << "\nChebyshev iteration method" << std::endl;
    test_Chebyshev(m, n);
}
