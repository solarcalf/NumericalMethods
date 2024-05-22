#ifndef __DIRICHLET_PROBLEM_HPP__
#define __DIRICHLET_PROBLEM_HPP__

#include "Solvers.hpp"
#include <vector>
#include <array>
#include <memory>
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
*       |__________________                  |
*                         |                  |
*                         |                  |
*                         |                  |
*                         |                  |
*                         |                  |
*                         |                  |
*                         |                  |
*                         |__________________|
*                     (x_n/2, y_0)       (x_n, y_0)  
*   
*/


namespace numcpp {


    enum GridType
    {
        Regular,
        ReversedR
    };


    template <GridType>
    class DirichletProblemSolver
    {
        using one_dim_function = std::function<FP(FP)>;
        using two_dim_function = std::function<FP(FP, FP)>;

        size_t m, n;
        std::unique_ptr<ISolver> solver;
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

        void set_solver(std::unique_ptr<ISolver> new_solver)
        {
            this->solver = std::move(new_solver);
        }

        void set_corners(std::array<FP, 4> new_corners)
        {
            this->corners = std::move(new_corners);
        }

        void set_u(two_dim_function new_u)
        {
            this->u = std::move(new_u);
        }

        void set_f(two_dim_function new_f)
        {
            this->f = std::move(new_f);
        }

        void set_boundary_conditions(std::array<one_dim_function, 4> new_mu)
        {
            this->mu1 = std::move(new_mu[0]);
            this->mu2 = std::move(new_mu[1]);
            this->mu3 = std::move(new_mu[2]);
            this->mu4 = std::move(new_mu[3]);
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
            MatrixForRegularGrid(FP h, FP k) : horizonal_coef(1 / (h * h)), vertical_coef(1 / (k * k)) {};

            FP at(size_t i, size_t j) override { return 3.14; }

            std::vector<FP> operator*(const std::vector<FP>&) override { return std::vector<FP>(5, 3.14); }
        };

        // ...

        std::vector<FP> b;

        // How it supposed to be
        std::vector<FP> initial_approximation{ 0, 0, 0 };
        std::unique_ptr<IMatrix> matrix = std::make_unique<MatrixForRegularGrid>(0.1, 0.1);
        MinRes solver(std::move(initial_approximation), 1000, 0.001, std::move(matrix), b);

        // Or this way
        MinRes solver2(std::move(initial_approximation), 1000, 0.001, std::make_unique<MatrixForRegularGrid>(0.1, 0.1), b);

        std::vector<FP> solution = solver.solve();

        // Create return value with std::vector<std::vector<FP>> type with obtained solution instead placeholder
        return std::vector<std::vector<FP>>();
    }


    // Implementation for r-shaped grid
    template <>
    std::vector<std::vector<FP>> const DirichletProblemSolver<GridType::ReversedR>::solve()
    {
        class MatrixForRShapedGrid : public numcpp::IMatrix
        {
            FP horizonal_coef, vertical_coef, A;  // Coefficient for cross template
            size_t n, m;

        public:
            MatrixForRShapedGrid(FP h, FP k, size_t n, size_t m) : horizonal_coef(1 / (h * h)), vertical_coef(1 / (k * k)), 
                A(-2 * (horizonal_coef + vertical_coef)) {
                this->n = n;
                this->m = m;
            };

            FP at(size_t i, size_t j) override {
                if (i == j)
                    return A;

                size_t boundary_1 = n / 2 - 1;
                size_t boundary_2 = (m / 2 - 1) * (n / 2 - 1);
                size_t boundary_3 = (m / 2) * (n / 2 - 1);
                size_t boundary_4 = (m / 2) * (n / 2 - 1) + n / 2 + 1;
                size_t boundary_5 = (m / 2 - 2) * (n - 1) + (m / 2) * (n / 2 - 1);

                if (i < boundary_1)
                {
                    if (((j != 0) && (i == j - 1) && (i + 1 == j)) || ((i != 0) && (i - 1 == j) && (i == j + 1)))
                    {
                        if (j != boundary_1)
                            return horizonal_coef;
                        else
                            return 0;
                    }
                    else if (j == i + n / 2 - 1)
                        return vertical_coef;
                    else
                        return 0;
                }
                else if (i < boundary_2) {
                    if (((i == j - 1) && (i + 1 == j)) || ((i - 1 == j) && (i == j + 1)))
                    {
                        if (!((i + 1) % (n / 2 - 1) == 0 && (j > i)) && !((j + 1) % (n / 2 - 1) == 0 && (i > j)))
                            return horizonal_coef;
                        else
                            return 0;
                    }
                    else if ((j == i + n / 2 - 1) || (j == i + 1 - n / 2))
                        return vertical_coef;
                    else
                        return 0;
                }
                else if (i < boundary_3) {
                    if (((i == j - 1) && (i + 1 == j)) || ((i - 1 == j) && (i == j + 1)))
                    {
                        if (!((i + 1) % (n / 2 - 1) == 0 && (j > i)) && !((j + 1) % (n / 2 - 1) == 0 && (i > j)))
                            return horizonal_coef;
                        else
                            return 0;
                    }
                    else if ((j == i + n - 1) || (j == i + 1 - n / 2))
                        return vertical_coef;
                    else
                        return 0;
                }
                else if (i < boundary_4) {
                    if (((i == j - 1) && (i + 1 == j)) || ((i - 1 == j) && (i == j + 1)))
                    {
                        if (j != boundary_3 - 1)
                            return horizonal_coef;
                        else
                            return 0;
                    }
                    else if (j == i + n - 1)
                        return vertical_coef;
                    else
                        return 0;
                }
                else if (i < boundary_5) {
                    if ((((i == j - 1) && (i + 1 == j)) || ((i - 1 == j) && (i == j + 1))))
                    {
                        if (!((i + 1 - boundary_3) % (n - 1) == 0 && (j > i)) && !((j + 1 - boundary_3) % (n - 1) == 0 && (i > j)))
                            return horizonal_coef;
                        else
                            return 0;
                    }
                    else if ((j == i + n - 1) || (j == i + 1 - n))
                        return vertical_coef;
                    else
                        return 0;
                }
                else {
                    if (((i != (n - 1)) && (i == j - 1) && (i + 1 == j) || (j != (m - 1)) && (i - 1 == j) && (i == j + 1)))
                    {
                        if (j != boundary_5 - 1)
                            return horizonal_coef;
                        else
                            return 0;
                    }
                    else if (j == i + 1 - n)
                        return vertical_coef;
                    else
                        return 0;
                }
            }

            std::vector<FP> operator*(const std::vector<FP>& vec) override {

                size_t vec_size = vec.size();

                std::vector<FP> result(vec_size, 0.0);

                result[0] = A * vec[0] + horizonal_coef * vec[1] + vertical_coef * vec[n / 2 - 1];
                for (size_t i = 1; i < n / 2 - 2; ++i)
                {
                    result[i] = horizonal_coef * vec[i - 1] + A * vec[i] + horizonal_coef * vec[i + 1] + vertical_coef * vec[i + n / 2 - 1];
                }
                result[n / 2 - 2] = horizonal_coef * vec[n / 2 - 3] + A * vec[n / 2 - 2] + vertical_coef * vec[n - 3];

                for (size_t i = 1; i < (m / 2) - 1; ++i)
                {
                    size_t ind = i * (n / 2 - 1);
                    result[ind] = vertical_coef * vec[ind - n / 2 + 1] + A * vec[ind] + horizonal_coef * vec[ind + 1] + vertical_coef * vec[ind + n / 2 - 1];
                    for (size_t j = 1; j < n / 2 - 2; ++j)
                    {
                        ind = i * (n / 2 - 1) + j;
                        result[ind] = vertical_coef * vec[ind - n / 2 + 1] + horizonal_coef * vec[ind - 1] + A * vec[ind] + horizonal_coef * vec[ind + 1] + vertical_coef * vec[ind + n / 2 - 1];
                    }
                    ind = i * (n / 2 - 1) + n / 2 - 2;
                    result[ind] = vertical_coef * vec[ind - n / 2 + 1] + horizonal_coef * vec[ind - 1] + A * vec[ind] + vertical_coef * vec[ind + n / 2 - 1];
                }

                size_t ind = ((m / 2) - 1) * (n / 2 - 1);
                result[ind] = vertical_coef * vec[ind - n / 2 + 1] + A * vec[ind] + horizonal_coef * vec[ind + 1] + vertical_coef * vec[ind + n - 1];
                for (size_t i = ind + 1; i < ind + n / 2 - 2; ++i)
                {
                    result[i] = vertical_coef * vec[i - n / 2 + 1] + horizonal_coef * vec[i - 1] + A * vec[i] + horizonal_coef * vec[i + 1] + vertical_coef * vec[i + n - 1];
                }
                ind = ((m / 2) - 1) * (n / 2 - 1) + n / 2 - 2;
                result[ind] = vertical_coef * vec[ind - n / 2 + 1] + horizonal_coef * vec[ind - 1] + A * vec[ind] + vertical_coef * vec[ind + n - 1];

                ind++;
                result[ind] = A * vec[ind] + horizonal_coef * vec[ind + 1] + vertical_coef * vec[ind + n - 1];
                for (size_t i = 1; i < n / 2; ++i)
                {
                    result[ind + i] = horizonal_coef * vec[ind + i - 1] + A * vec[ind + i] + horizonal_coef * vec[ind + i + 1] + vertical_coef * vec[ind + i + n - 1];
                }
                ind += n / 2;
                for (size_t i = 0; i < n / 2 - 2; ++i)
                {
                    result[ind + i] = vertical_coef * vec[ind + i - n + 1] + horizonal_coef * vec[ind + i - 1] + A * vec[ind + i] + horizonal_coef * vec[ind + i + 1] + vertical_coef * vec[ind + i + n - 1];
                }
                ind += (n / 2 - 2);
                result[ind] = vertical_coef * vec[ind - n + 1] + horizonal_coef * vec[ind - 1] + A * vec[ind] + vertical_coef * vec[ind + n - 1];

                ind++;
                for (size_t i = 0; i < (m / 2) - 3; ++i)
                {
                    size_t ind_1 = ind + i * (n - 1);
                    result[ind_1] = vertical_coef * vec[ind_1 - n + 1] + A * vec[ind_1] + horizonal_coef * vec[ind_1 + 1] + vertical_coef * vec[ind_1 + n - 1];
                    for (size_t j = 1; j < n - 2; ++j)
                    {
                        result[ind_1 + j] = vertical_coef * vec[ind_1 + j - n + 1] + horizonal_coef * vec[ind_1 + j - 1] + A * vec[ind_1 + j] + horizonal_coef * vec[ind_1 + j + 1] + vertical_coef * vec[ind_1 + j + n - 1];

                    }
                    ind_1 = ind + i * (n - 1) + n - 2;
                    result[ind_1] = vertical_coef * vec[ind_1 - n + 1] + horizonal_coef * vec[ind_1 - 1] + A * vec[ind_1] + vertical_coef * vec[ind_1 + n - 1];
                }

                ind = ind + ((m / 2) - 3) * (n - 1);
                result[ind] = vertical_coef * vec[ind - n + 1] + A * vec[ind] + horizonal_coef * vec[ind + 1];
                for (size_t i = 1; i < n - 2; ++i)
                {
                    result[ind + i] = vertical_coef * vec[ind + i - n + 1] + horizonal_coef * vec[ind + i - 1] + A * vec[ind + i] + horizonal_coef * vec[ind + i + 1];
                }
                ind = ind + n - 2;
                result[ind] = vertical_coef * vec[ind - n + 1] + horizonal_coef * vec[ind - 1] + A * vec[ind];

                return result;
            }
        };

        // Placeholder
        return std::vector<std::vector<FP>>();
    }


}  // namespace numcpp



#endif // __DIRICHLET_PROBLEM_HPP__
