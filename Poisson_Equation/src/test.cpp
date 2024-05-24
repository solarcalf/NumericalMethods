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

class mat : public numcpp::IMatrix
{
public:
    std::vector<std::vector<FP>> m{
        {4, 0, 3},
        {0, 2, -1},
        {3, -1, -2}
    };
        //std::vector<std::vector<FP>> m{
    //{15.0, 9.0, 2.25, 0},
    //{9, 15 , 0, 2.25},
    //{2.25, 0, 15, 9},
    //{0, 2.25, 9, 15}
    //}; //чебышева тест

    ~mat() override = default;

    FP at(size_t i, size_t j) override {
        return m[i][j];
    }

    std::vector<FP> operator*(const std::vector<FP>& v) override {
        std::vector<FP> res(v.size());

        for (size_t row = 0; row < m.size(); ++row) 
            for (size_t k = 0; k < v.size(); ++k) 
                res[row] += m[row][k] * v[k];
        
        return res;
    }
    FP size() override {
        return m.size();
    }

};
void test_TopRelaxation(){
    size_t n = 2048;
    size_t m = 2048;
    FP omega = 1.95;
    std::cout << "n = " << n << " m = " << m << '\n';
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
    dirichlet_task.set_solver(std::make_unique<numcpp::TopRelaxationOptimizedForDirichletRegularGrid>(init_app, 1000000000, 0.0000000000001, nullptr, std::vector<FP>(),
     f, mu1, mu2, mu3, mu4, n, m, corners, omega));
    auto start = std::chrono::high_resolution_clock::now();
    
    auto solution = dirichlet_task.solve();

    auto end = std::chrono::high_resolution_clock::now();
    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Время работы в секундах: " << duration / 1000000 << std::endl;
    // for (const auto& row : solution) {
    //     for (double val : row) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;
    // }
}
void test_ConGrad()
{
    size_t n = 400;
    size_t m = 400;

    size_t sz = (n / 2 - 1) * (m - 1) + (n / 2) * (m / 2 - 1);

    std::array<double, 4> corners = {-1.0, -1.0, 1.0, 1.0};
    auto u = [](double x, double y) { return exp(1 - pow(x, 2) - pow(y, 2)); };
    auto f = [](double x, double y) { return -4 * exp(1 - pow(x, 2) - pow(y, 2)) * (pow(y, 2) + pow(x, 2) - 1); };
    auto mu1 = [](double y) { return exp(-pow(y, 2)); };
    auto mu2 = [](double y) { return exp(1.0 - pow(y, 2)); };
    auto mu3 = [](double y) { return exp(-pow(y, 2)); };
    auto mu4 = [](double x) { return exp(-pow(x, 2)); };
    auto mu5 = [](double x) { return exp(1.0 - pow(x, 2)); };
    auto mu6 = [](double x) { return exp(-pow(x, 2)); };

    std::vector<FP> initial_approximation(sz, 0.0);
    
    numcpp::DirichletProblemSolver<numcpp::GridType::ReversedR> solver;

    solver.set_solver(std::make_unique<numcpp::ConGrad>(initial_approximation, 10000, 0.000000001, nullptr, std::vector<FP>()));

    solver.set_fraction(n, m);
    solver.set_corners(corners);
    solver.set_u(u);
    solver.set_f(f);
    std::array<std::function<FP(FP)>, 6> arr{mu1, mu2, mu3, mu4, mu5, mu6};
    solver.set_boundary_conditions_for_r_shaped_grid(arr);

    auto start = std::chrono::high_resolution_clock::now();
    solver.solve();
    auto end = std::chrono::high_resolution_clock::now();
    int64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Время работы в секундах: " << duration / 1000000 << std::endl;
}

int main() {
    test_ConGrad();
    // mat m;
    // std::vector<FP> b{ 13, 1, -5 };

    // std::vector<FP> initial_approximation{ 0, 0, 0 };
    // std::unique_ptr<numcpp::IMatrix> matrix = std::make_unique<mat>();

    //numcpp::MinRes solver({ 0, 0, 0 }, 1000, 0.00001, std::move(matrix), b);
    //auto res = solver.solve();
    //auto res = solver.solve();
    //for (auto val : res)
    //    std::cout << val << ' ';
    //std::cout << '\n';


    // numcpp::ChebyshevIteration solver_cheb({ 0, 0, 0 }, 1000, 0.00001, std::move(matrix), b);
    // FP Mmin = -6;
    // FP Mmax = 7;
    // solver_cheb.set_Mmin(Mmin);
    // solver_cheb.set_Mmax(Mmax);
    // auto res_cheb = solver_cheb.solve();
    // for (auto val : res_cheb)
    //     std::cout << val << ' ';
    // std::cout << '\n';

    //mat m;
    //std::vector<FP> b{ 26.25, 26.25, 26.25, 26.25 };
    //std::vector<FP> initial_approximation{ 0, 0, 0, 0 };
    //std::unique_ptr<numcpp::IMatrix> matrix = std::make_unique<mat>();
    //FP size = (*matrix).size();
    //FP k = 0.66;
    //FP h = 0.33;
    //FP Mmin = 4.0 / pow(h, 2) * pow(sin(PI / 2.0 / (size + 1)), 2) + 4.0 / pow(k, 2) * pow(sin(PI / 2.0 / (size + 1)), 2);
    //FP Mmax = 4.0 / pow(h, 2) * pow(sin(PI * (size) / 2.0 / (size + 1)), 2) + 4.0 / pow(k, 2) * pow(sin(PI * (size) / 2.0 / (size + 1)), 2);
    //numcpp::ChebyshevIteration solver_cheb({ 0, 0, 0, 0 }, 1000, 0.00001, std::move(matrix),b);
    //solver_cheb.set_Mmin(Mmin);
    //solver_cheb.set_Mmax(Mmax);
    //auto res_cheb = solver_cheb.solve();
    //for (auto val : res_cheb)
    //    std::cout << val << ' ';
    //std::cout << '\n';
}
