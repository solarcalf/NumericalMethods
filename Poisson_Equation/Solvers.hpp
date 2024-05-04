#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#include <vector>
#include <memory>

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
protected:
    std::vector<FP> initial_approximation;
    size_t max_iterations;
    FP required_precision;
    std::unique_ptr<IMatrix> system_matrix;

    ISolver(std::vector<FP> init_approx, size_t max_iters, FP required_precision, std::unique_ptr<IMatrix> system_matrix)
        : initial_approximation(std::move(init_approx)),
          max_iterations(max_iters),
          required_precision(required_precision),
          system_matrix(std::move(system_matrix)) {}

public:
    virtual std::vector<FP> solve() const = 0;

    virtual ~ISolver();

    ISolver() = default;

    void set_initial_approximation(const std::vector<FP>& init_approx)
    {
        this->initial_approximation = init_approx;
    }

    void set_initial_approximation(std::vector<FP>&& init_approx)
    {
        this->initial_approximation = std::move(init_approx);
    }

    void set_max_iterations(size_t max_iters)
    {
        this->max_iterations = max_iters;
    }

    void set_required_precision(FP required_precision)
    {
        this->required_precision = required_precision;
    }

    void set_system_matrix(std::unique_ptr<IMatrix> system_matrix)
    {
        this->system_matrix = std::move(system_matrix);
    }
};


// Minimal residual method
class MinRes: public ISolver 
{
public:
    MinRes(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix):
    ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix)) {}

    ~MinRes() = default;

    std::vector<FP> solve() const override;
};


// Chebyshev iteration method
class ChebyshevIteration: public ISolver
{
public:
    ChebyshevIteration(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix):
    ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix)) {}

    ~ChebyshevIteration() = default;

    std::vector<FP> solve() const override;
};
    

} // namespace numcpp


#endif // __SOLVER_HPP__
