#ifndef __DIRICHLET_PROBLEM_HPP__
#define __DIRICHLET_PROBLEM_HPP__

#include "Solvers.hpp"
#include <vector>
#include <array>
#include <memory>
#include <iostream>
#include <functional>
#include <fstream>

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
        size_t task_num;
        std::unique_ptr<ISolver> solver;
        std::array<FP, 4> corners;  // {x0, y0, xn, ym}

        two_dim_function u, f;
        one_dim_function mu1, mu2, mu3, mu4, mu5, mu6;

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

        void set_boundary_conditions_for_r_shaped_grid(std::array<one_dim_function, 6> new_mu)
        {
            this->mu1 = std::move(new_mu[0]);
            this->mu2 = std::move(new_mu[1]);
            this->mu3 = std::move(new_mu[2]);
            this->mu4 = std::move(new_mu[3]);
            this->mu5 = std::move(new_mu[4]);
            this->mu6 = std::move(new_mu[5]);
        }

        void set_task_num(size_t num)
        {
            this->task_num = num;
        }
    };


    // Implementation for regular grid
template<>
std::vector<std::vector<FP>> const DirichletProblemSolver<GridType::Regular>::solve()
{
    class MatrixForRegularGrid : public numcpp::IMatrix
    {
        FP horizonal_coef, vertical_coef, A;  // Coefficient for cross template
        size_t n, m;
    public:
        MatrixForRegularGrid(size_t n, size_t m, FP h, FP k):n(n), m(m),
         horizonal_coef(1.0 / (h * h)), vertical_coef(1.0 / (k * k)), A(-2*(horizonal_coef + vertical_coef))
        {};

        FP at(size_t i, size_t j) override 
        {
            if(i == j)
                return A;
            else if(i == j + n - 1 || i + n - 1 == j)
                return vertical_coef;
            else if(i == j + 1 && i % (n - 1) != 0 || i + 1 == j && j % (n - 1) != 0)
                return horizonal_coef;
            else return 0;

        };

        std::vector<FP> operator*(const std::vector<FP>& v) override 
        {
            std::vector<FP> result(v.size());

            //first block row
            result[0] = A * v[0] + horizonal_coef * v[1] + vertical_coef * v[n - 1];
#pragma omp parallel for
            for(size_t i = 1; i < n - 2; ++i)
            {
                result[i] = horizonal_coef * v[i - 1] + A * v[i] + horizonal_coef * v[i + 1] + vertical_coef * v[i + n - 1];
            }
            result[n - 2] = horizonal_coef * v[n - 3] + A * v[n - 2] + vertical_coef * v[2 * n - 3];

            //main part
#pragma omp parallel for
            for(size_t i = 1; i < m - 1; ++i)
            {
                size_t ind = i * (n - 1);
                result[ind] = vertical_coef * v[ind - n + 1] + A * v[ind] + horizonal_coef * v[ind + 1] + vertical_coef * v[ind + n - 1];
                for(size_t j = 1; j < n - 2; ++j)
                {   ind++;
                    result[ind] = vertical_coef * v[ind - n + 1] + horizonal_coef * v[ind - 1] + A * v[ind] + horizonal_coef * v[ind + 1] + vertical_coef * v[ind + n - 1];
                } 
                ind++;
                result[ind] = vertical_coef * v[ind - n + 1] + horizonal_coef * v[ind - 1] + A * v[ind] + vertical_coef * v[ind + n - 1];
            }

            //last block row
            size_t ind = (n - 1) * (m - 1) - n + 1;
            result[ind] = vertical_coef * v[ind - n + 1] + A * v[ind] + horizonal_coef * v[ind + 1];
#pragma omp parallel for
            for(size_t i = 1; i < n - 2; ++i)
            {
                size_t ind_for = (n - 1) * (m - 1) - n + 1 + i;
                result[ind_for] = vertical_coef * v[ind_for - n + 1] + horizonal_coef * v[ind_for - 1] + A * v[ind_for] + horizonal_coef * v[ind_for + 1];
            }
            ind = (n - 1) * (m - 1) - 1;
            result[ind] = vertical_coef * v[ind - n + 1] + horizonal_coef * v[ind - 1] + A * v[ind];

            return result;
        };
        FP size(){ return (n - 1) * (m - 1); }
    };

    std::vector<FP> b((n - 1)*(m - 1));
    FP h, k, horizonal_coef, vertical_coef;
    h = (corners[2] - corners[0]) / n;
    k = (corners[3] - corners[1]) / m;
    horizonal_coef = 1 / (h * h);
    vertical_coef =  1 / (k * k);

    for(size_t j = 0; j < m - 1; j++)
        for(size_t i = 0; i < n - 1; i++)
        {
            size_t ind = (n-1)*j + i;
            FP y = corners[1] + (j+1) * k;
            FP x = corners[0] + (i+1) * h;
            b[ind] = -f(x, y);

            if(j == 0)
                b[ind] -= mu3(x)*vertical_coef;
            
            else if(j == m - 2)
                b[ind] -= mu4(x)*vertical_coef;
            
            if(i == 0)
                b[ind] -= mu1(y)*horizonal_coef;
            
            else if(i == n - 2)
                b[ind] -= mu2(y)*horizonal_coef;
            
        }

    std::unique_ptr<IMatrix> matrix = std::make_unique<MatrixForRegularGrid>(n, m, h, k);
    solver->set_system_matrix(std::move(matrix));
    solver->set_b(b);
    std::vector<FP> solution = solver->solve();
    std::vector<std::vector<FP>> res(n + 1, std::vector<FP>(m + 1, 0));
    if(task_num == 0)
    {
        FP error = 0; 
        FP x_max = 0;
        FP y_max = 0;

        for(size_t i = 0; i < m - 1; i++)
        {
            for(size_t j = 0; j < n - 1; j++)
            {
                FP y = corners[1] + (i + 1) * k;
                FP x = corners[0] + (j + 1) * h;
                FP dot_solution = solution[(n - 1) * i + j];
                FP dot_correct = u(x, y);
                if(std::abs(dot_correct - dot_solution) > error)
                {
                    error = std::abs(dot_correct - dot_solution);
                    x_max = x;
                    y_max = y;
                }
            } 
        }
        std::ofstream fout_res("../files/Error.txt");
        fout_res << "Задача решена с погрешностью " << error << "\nМаксимальное отклонение точного и численного решений в точке x = " << x_max << ", y = " << y_max << '\n';
        fout_res.close();
    }
    


    FP start_x = corners[0];
    FP start_y = corners[1];

    for (size_t j = 0; j <= m; ++j)
    {
        res[0][j] = mu1(start_y + static_cast<FP>(j) * k);
    }
    for (size_t j = 0; j <= m; ++j)
    {
        res[n][j] = mu2(start_y + static_cast<FP>(j) * k);
    }


    for (size_t i = 1; i <= n; ++i)
    {
        res[i][0] = mu3(start_x + static_cast<FP>(i) * h);
    }
    for (size_t i = 1; i <= n; ++i)
    {
        res[i][m] = mu4(start_x + static_cast<FP>(i) * h);
    }

    size_t index = 0;
    for (size_t j = 1; j < m; ++j)
    {
        for (size_t i = 1; i < n; ++i)
        {
            res[i][j] = solution[index++];
        }
    }



    return res;
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

            FP size() override {
                return (n / 2 - 1) * (m - 1) + (n / 2) * (m / 2 - 1);
            }

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
#pragma omp parallel for
                for (size_t i = 1; i < n / 2 - 2; ++i)
                {
                    result[i] = horizonal_coef * vec[i - 1] + A * vec[i] + horizonal_coef * vec[i + 1] + vertical_coef * vec[i + n / 2 - 1];
                }
                result[n / 2 - 2] = horizonal_coef * vec[n / 2 - 3] + A * vec[n / 2 - 2] + vertical_coef * vec[n - 3];
#pragma omp parallel for
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
#pragma omp parallel for
                for (size_t i = ind + 1; i < ind + n / 2 - 2; ++i)
                {
                    result[i] = vertical_coef * vec[i - n / 2 + 1] + horizonal_coef * vec[i - 1] + A * vec[i] + horizonal_coef * vec[i + 1] + vertical_coef * vec[i + n - 1];
                }
                ind = ((m / 2) - 1) * (n / 2 - 1) + n / 2 - 2;
                result[ind] = vertical_coef * vec[ind - n / 2 + 1] + horizonal_coef * vec[ind - 1] + A * vec[ind] + vertical_coef * vec[ind + n - 1];

                ind++;
                result[ind] = A * vec[ind] + horizonal_coef * vec[ind + 1] + vertical_coef * vec[ind + n - 1];
#pragma omp parallel for
                for (size_t i = 1; i < n / 2; ++i)
                {
                    result[ind + i] = horizonal_coef * vec[ind + i - 1] + A * vec[ind + i] + horizonal_coef * vec[ind + i + 1] + vertical_coef * vec[ind + i + n - 1];
                }
                ind += n / 2;
#pragma omp parallel for
                for (size_t i = 0; i < n / 2 - 2; ++i)
                {
                    result[ind + i] = vertical_coef * vec[ind + i - n + 1] + horizonal_coef * vec[ind + i - 1] + A * vec[ind + i] + horizonal_coef * vec[ind + i + 1] + vertical_coef * vec[ind + i + n - 1];
                }
                ind += (n / 2 - 2);
                result[ind] = vertical_coef * vec[ind - n + 1] + horizonal_coef * vec[ind - 1] + A * vec[ind] + vertical_coef * vec[ind + n - 1];

                ind++;
#pragma omp parallel for
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
#pragma omp parallel for
                for (size_t i = 1; i < n - 2; ++i)
                {
                    result[ind + i] = vertical_coef * vec[ind + i - n + 1] + horizonal_coef * vec[ind + i - 1] + A * vec[ind + i] + horizonal_coef * vec[ind + i + 1];
                }
                ind = ind + n - 2;
                result[ind] = vertical_coef * vec[ind - n + 1] + horizonal_coef * vec[ind - 1] + A * vec[ind];

                return result;
            }
        };

        FP h = (corners[2] - corners[0]) / n;
        FP k = (corners[3] - corners[1]) / m;
        size_t sz = (n / 2 - 1) * (m - 1) + (n / 2) * (m / 2 - 1);

        FP horizonal_coef = 1 / (h * h);
        FP vertical_coef = 1 / (k * k);

        std::vector<FP> b;
        std::vector<FP> real_sol;

        FP start_x = corners[0];
        FP start_y = corners[1];

        for (size_t j = 1; j <= m / 2; ++j)
        {
            for (size_t i = n / 2 + 1; i < n; ++i)
            {
                b.push_back(-f(start_x + static_cast<FP>(i) * h, start_y + static_cast<FP>(j) * k));
                real_sol.push_back(u(start_x + static_cast<FP>(i) * h, start_y + static_cast<FP>(j) * k));
            }
        }
        for (size_t j = m / 2 + 1; j < m; ++j)
        {
            for (size_t i = 1; i < n; ++i)
            {
                b.push_back(-f(start_x + static_cast<FP>(i) * h, start_y + static_cast<FP>(j) * k));
                real_sol.push_back(u(start_x + static_cast<FP>(i) * h, start_y + static_cast<FP>(j) * k));
            }
        }

        b[0] -= horizonal_coef * mu2(start_y + k) + vertical_coef * mu4(start_x + static_cast<FP>(n / 2 + 1) * h);
        for (size_t i = 1; i < n / 2 - 2; ++i)
        {
            b[i] -= vertical_coef * mu4(start_x + static_cast<FP>(n / 2 + 1 + i) * h);
        }
        b[n / 2 - 2] -= horizonal_coef * mu3(start_y + k) + vertical_coef * mu4(start_x + static_cast<FP>(n - 1) * h);
        
        size_t ind = 0;
        for (size_t i = 1; i < (m / 2); ++i)
        {
            ind = i * (n / 2 - 1);
            b[ind] -= horizonal_coef * mu2(start_y + static_cast<FP>(i + 1) * k);
            ind = i * (n / 2 - 1) + n / 2 - 2;
            b[ind] -= horizonal_coef * mu3(start_y + static_cast<FP>(i + 1) * k);
        }

        ind++;
        b[ind] -= horizonal_coef * mu1(start_y + static_cast<FP>(m / 2 + 1) * k) + vertical_coef * mu5(start_x + h);
        for (size_t i = 1; i < n / 2; ++i)
        {
            b[ind + i] -= vertical_coef * mu5(start_x + static_cast<FP>(i + 1) * h);
        }
        ind += (n - 2);
        b[ind] -= horizonal_coef * mu3(start_y + static_cast<FP>(m / 2 + 1) * k);

        ind++;
        for (size_t i = 0; i < (m / 2) - 3; ++i)
        {
            size_t ind_1 = ind + i * (n - 1);
            b[ind_1] -= horizonal_coef * mu1(start_y + static_cast<FP>(m / 2 + 2 + i) * k);
            ind_1 = ind + i * (n - 1) + n - 2;
            b[ind_1] -= horizonal_coef * mu3(start_y + static_cast<FP>(m / 2 + 2 + i) * k);
        }

        ind = ind + ((m / 2) - 3) * (n - 1);
        b[ind] -= vertical_coef * mu6(start_x + h) + horizonal_coef * mu1(start_y + static_cast<FP>(m - 1) * k);
        for (size_t i = 1; i < n - 2; ++i)
        {
            b[ind + i] -= vertical_coef * mu6(start_x + static_cast<FP>(i + 1) * h);
        }
        ind = ind + n - 2;
        b[ind] -= vertical_coef * mu6(start_x + static_cast<FP>(n - 1) * h) + horizonal_coef * mu3(start_y + static_cast<FP>(m - 1) * k);

        std::unique_ptr<IMatrix> matrix = std::make_unique<MatrixForRShapedGrid>(h, k, n, m);
        solver->set_system_matrix(std::move(matrix));
        solver->set_b(b);
        std::vector<FP> solution = solver->solve();
        
        FP x_max_err = 0.0;
        FP y_max_err = 0.0;
        FP approximation_error = 0.0;
        for (size_t i = 0; i < sz; ++i)
        {
            if (std::abs(solution[i] - real_sol[i]) > approximation_error)
            {
                approximation_error = std::abs(solution[i] - real_sol[i]);
                if (i < (n / 2 - 1) * (m / 2))
                {
                    x_max_err = start_x + static_cast<FP>(n / 2 + 1 + i % (n / 2 - 1)) * h;
                    y_max_err = start_y + static_cast<FP>(1 + static_cast<int>(i / (n / 2 - 1))) * k;
                }
                else
                {
                    x_max_err = start_x + static_cast<FP>(1 + (i - (n / 2 - 1) * (m / 2)) % (n - 1)) * h;
                    y_max_err = start_y + static_cast<FP>(m / 2 + 1 + static_cast<int>((i - (n / 2 - 1) * (m / 2)) / (n - 1))) * k;
                }
            }
        }

        std::cout << "Node with maximum error: x = " << x_max_err << ", y = " << y_max_err << std::endl;
        std::cout << "General error: " << approximation_error << std::endl;

        std::ofstream fout_res("../files/Error.txt");
        fout_res << "Задача решена с погрешностью " << approximation_error << "\nМаксимальное отклонение точного и численного решений в точке x = " << x_max_err << ", y = " << y_max_err << '\n';
        fout_res.close();

        std::vector<std::vector<FP>> res(n + 1, std::vector<FP>(m + 1, 0));

        for (size_t j = m / 2; j <= m; ++j)
        {
            res[0][j] = mu1(start_y + static_cast<FP>(j) * k);
        }
        for (size_t j = 0; j <= m / 2; ++j)
        {
            res[n / 2][j] = mu2(start_y + static_cast<FP>(j) * k);
        }
        for (size_t j = 0; j <= m; ++j)
        {
            res[n][j] = mu3(start_y + static_cast<FP>(j) * k);
        }

        for (size_t i = n / 2 + 1; i < n; ++i)
        {
            res[i][0] = mu4(start_x + static_cast<FP>(i) * h);
        }
        for (size_t i = 1; i < n / 2; ++i)
        {
            res[i][m / 2] = mu5(start_x + static_cast<FP>(i) * h);
        }
        for (size_t i = 1; i < n; ++i)
        {
            res[i][m] = mu6(start_x + static_cast<FP>(i) * h);
        }

        size_t index = 0;
        for (size_t j = 1; j <= m / 2; ++j)
        {
            for (size_t i = n / 2 + 1; i < n; ++i)
            {
                res[i][j] = solution[index++];
            }
        }
        for (size_t j = m / 2 + 1; j < m; ++j)
        {
            for (size_t i = 1; i < n; ++i)
            {
                res[i][j] = solution[index++];
            }
        }

        // Placeholder
        return res;
    }


}  // namespace numcpp



#endif // __DIRICHLET_PROBLEM_HPP__