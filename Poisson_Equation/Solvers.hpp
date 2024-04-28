#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#include <vector>

using FP = double;


// Abstract class to provide interface and utility methods
class Solver {
private:
    std::vector<std::vector<FP>> system_matrix;
    std::vector<FP> initital_approximation;
    size_t max_iterations;
    FP required_precision;

public:
    virtual ~Solver() = 0;

};


// Minimal residual method
class MinRes: public Solver{};


// Chebyshev iteration method
class ChebyshevIteration: public Solver{};


#endif // __SOLVER_HPP__
