#ifndef __DIRICHLET_PROBLEM_HPP__
#define __DIRICHLET_PROBLEM_HPP__

#include "Solvers.hpp"
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
public:
    using one_dim_function = std::function<FP(FP)>;
    using two_dim_function = std::function<FP(FP, FP)>;

private:
    size_t m, n;
    Solver& solver;
    std::array<FP, 4> corners; // {x0, y0, xn, ym}

    two_dim_function u, f;
    one_dim_function mu1, mu2, mu3, mu4;

 public:   
    void solve();
};


// Implementation for regular grid
template <>
void DirichletProblemSolver<GridType::Regular>::solve() 
{

}

// Implementation for r-shaped grid
template <>
void DirichletProblemSolver<GridType::Shape_r>::solve() 
{

}


} // namespace numcpp


#endif // __DIRICHLET_PROBLEM_HPP__
