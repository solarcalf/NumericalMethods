#include <vector>
#include <algorithm>
#include <memory>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cmath>

// #include "Dirichlet_Problem.hpp"
#include "Solvers.hpp"

#define FP double

class mat : public numcpp::IMatrix
{
public:
    std::vector<std::vector<FP>> m{
        {4, 0, 3},
        {0, 2, -1},
        {3, -1, -2}
    };

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

};

int main() {
    mat m;
    std::vector<FP> b{13, 1, -5};
    
    std::vector<FP> initial_approximation{0, 0, 0};
    std::unique_ptr<numcpp::IMatrix> matrix = std::make_unique<mat>();

    numcpp::MinRes solver({0, 0, 0}, 1000, 0.00001, std::move(matrix), b);
    auto res = solver.solve();

    for (auto val : res)
        std::cout << val << ' ';
    std::cout << '\n';
}