#ifndef __DIRICHLET_PROBLEM_HPP__
#define __DIRICHLET_PROBLEM_HPP__

#include "Solvers.hpp"
#include <vector>
#include <array>
#include <functional>

using FP = double;


/*
*   Outline of r-shaped grid:
*
*   (x_0, y_m)                           (x_n, y_m)
*       ______________________________________
*       |                                    |
*       |                                    |
*       |                                    |
*       |                                    |
*       |                                    |
*       |                                    |
*       |                  __________________|
*       |                  |
*       |                  |
*       |                  |
*       |                  |
*       |                  |
*       |                  |
*       |                  |
*       |__________________|
*   (x_0, y_0)       (x_n/2, y_0)  
*   
*/


namespace numcpp {


enum GridType 
{
    Regular,
    Shape_r
};


template <GridType>
class DirichletProblemSolver 
{
    using one_dim_function = std::function<FP(FP)>;
    using two_dim_function = std::function<FP(FP, FP)>;

    size_t m, n;
    ISolver& solver;
    std::array<FP, 4> corners;  // {x0, y0, xn, ym}

    two_dim_function u, f;
    one_dim_function mu1, mu2, mu3, mu4;

 public:   
    std::vector<std::vector<FP>> const solve();

    void set_fraction(size_t new_m, size_t new_n)
    {
        this->m = new_m;
        this->n = new_n;
    }

    void set_solver(ISolver& new_solver)
    {
        this->solver = new_solver;
    }

    void set_corners(std::array<FP, 4> new_corners)
    {
        this->corners = new_corners;
    }

    void set_u(two_dim_function new_u)
    {
        this->u = new_u;
    }

    void set_f(two_dim_function new_f)
    {
        this->f = new_f;
    }

    void set_boundary_conditions(std::array<one_dim_function, 4> new_mu)
    {
        this->mu1 = new_mu[0];
        this->mu2 = new_mu[1];
        this->mu3 = new_mu[2];
        this->mu4 = new_mu[3];
    }
};


// Implementation for regular grid
template<>
std::vector<std::vector<FP>> const DirichletProblemSolver<GridType::Regular>::solve()
{
    class MatrixForRegularGrid : public numcpp::IMatrix
    {
        FP horizonal_coef, vertical_coef;  // Coefficient for cross template

    public:
        MatrixForRegularGrid(FP h, FP k): horizonal_coef(1 / (h * h)), vertical_coef(1 / (k * k)) {};

        FP at(size_t i, size_t j) override;
    };

    // ...

    // How it supposed to be
    std::vector<FP> initial_approximation{0, 0, 0};
    MatrixForRegularGrid matrix(0.1, 0.1);
    MinRes solver(initial_approximation, 1000, 0.001, matrix);
    auto solution = solver.solve();

    // Create return value with std::vector<std::vector<FP>> type with obtained solution
}


// Implementation for r-shaped grid
template <>
std::vector<std::vector<FP>> const DirichletProblemSolver<GridType::Shape_r>::solve() 
{

}


}  // namespace numcpp


#endif // __DIRICHLET_PROBLEM_HPP__
