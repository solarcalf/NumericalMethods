#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#include <vector>

using FP = double;

namespace numcpp 
{


// Matrix interface
class IMatrix 
{
public:
    virtual FP at(size_t, size_t) = 0;
};


// Solver interface
class ISolver 
{
private:
    std::vector<FP> initital_approximation;
    size_t max_iterations;
    FP required_precision;
    IMatrix& system_matrix;

protected:
    ISolver(std::vector<FP> init_approx, size_t max_iters, FP required_precision, IMatrix& system_matrix): 
    initital_approximation(init_approx), max_iterations(max_iters), required_precision(required_precision), system_matrix(system_matrix) {};

public:
    virtual std::vector<FP> solve() const = 0;

    virtual ~ISolver();

    ISolver() = default;

    void set_initial_approximation(const std::vector<FP>& init_approx)
    {
        this->initital_approximation = init_approx;
    }

    void set_initial_approximation(std::vector<FP>&& init_approx)
    {
        this->initital_approximation = std::move(init_approx);
    }

    void set_max_iterations(size_t max_iters)
    {
        this->max_iterations = max_iters;
    }

    void set_required_precision(FP required_precision)
    {
        this->required_precision = required_precision;
    }

    void set_system_matrix(IMatrix& system_matrix)
    {
        this->system_matrix = system_matrix;
    }
};


// Minimal residual method
class MinRes: public ISolver 
{
public:
    MinRes(std::vector<FP> initital_approximation, size_t max_iterations, FP required_precision, IMatrix& system_matrix):
    ISolver(initital_approximation, max_iterations, required_precision, system_matrix) {}

    ~MinRes() = default;

    std::vector<FP> solve() const override;
};


// Chebyshev iteration method
class ChebyshevIteration: public ISolver
{
public:
    ChebyshevIteration(std::vector<FP> initital_approximation, size_t max_iterations, FP required_precision, IMatrix& system_matrix):
    ISolver(initital_approximation, max_iterations, required_precision, system_matrix) {}

    ~ChebyshevIteration() = default;

    std::vector<FP> solve() const override;
};
    

} // namespace numcpp


#endif // __SOLVER_HPP__
