#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#include <vector>
#include <algorithm>
#include <memory>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <functional>

#include "./Dirichlet_Problem.hpp"


using FP = double;

namespace numcpp 
{
    const double PI = 3.14159265358;

    FP scalar_product(const std::vector<FP>& lhs, const std::vector<FP>& rhs)
    {
        FP product = 0;
        const size_t size = lhs.size();
        const size_t end_of_vectorized_region = size - (size % 4);

#pragma omp parallel for reduction(+: product)
        for (size_t i = 0; i < end_of_vectorized_region; i += 4)
        {
            __m256d va = _mm256_loadu_pd(&lhs[i]);
            __m256d vb = _mm256_loadu_pd(&rhs[i]);
            __m256d vc = _mm256_mul_pd(va, vb);

            __m128d lo = _mm256_extractf128_pd(vc, 0);
            __m128d hi = _mm256_extractf128_pd(vc, 1);

            __m128d vs = _mm_add_pd(lo, hi);

            FP result[2];
            _mm_storeu_pd(result, vs);

            product += (result[0] + result[1]);
        }

        for (size_t i = end_of_vectorized_region; i < size; ++i)
            product += lhs[i] * rhs[i];

        return product;
    }

    FP norm(const std::vector<FP>& v)
    {
        FP norm = 0;

#pragma omp parallel for reduction(max: norm)
        for (size_t i = 0; i < v.size(); ++i)
            norm = std::max(norm, std::abs(v[i]));

        return norm;
    }

    FP error(const std::vector<FP>& v1, const std::vector<FP>& v2)
    {
        FP error = 0.0;

#pragma omp parallel for reduction(max: error)
        for (size_t i = 0; i < v1.size(); ++i)
        {
            error = std::max(std::abs(v1[i] - v2[i]), error);
        }
        return error;
    }

    std::vector<FP> vector_FMA(const std::vector<FP>& lhs, FP coef, const std::vector<FP>& rhs)
    {
        const size_t size = lhs.size();
        const size_t end_of_vectorized_region = size - (size % 4);
        std::vector<FP> result(size);

#pragma omp parallel for
        for (size_t i = 0; i < end_of_vectorized_region; i += 4)
        {
            __m256d va = _mm256_loadu_pd(&lhs[i]);
            __m256d vb = _mm256_loadu_pd(&rhs[i]);
            __m256d vc = _mm256_set1_pd(coef);
            __m256d vr = _mm256_fmadd_pd(va, vc, vb);
            _mm256_storeu_pd(&result[i], vr);
        }

        for (size_t i = end_of_vectorized_region; i < size; ++i)
            result[i] = lhs[i] * coef + rhs[i];

        return result;
    }

    std::vector<FP> operator-(const std::vector<FP>& lhs, const std::vector<FP>& rhs)
    {
        const size_t size = lhs.size();
        const size_t end_of_vectorized_region = size - (size % 4);
        std::vector<FP> result(size);

#pragma omp parallel for
        for (size_t i = 0; i < end_of_vectorized_region; i += 4)
        {
            __m256d va = _mm256_loadu_pd(&lhs[i]);
            __m256d vb = _mm256_loadu_pd(&rhs[i]);
            __m256d vc = _mm256_set1_pd(-1);
            __m256d vr = _mm256_fmadd_pd(vb, vc, va);
            _mm256_storeu_pd(&result[i], vr);
        }

        for (size_t i = end_of_vectorized_region; i < size; ++i)
            result[i] = std::fma(rhs[i], -1, lhs[i]);

        return result;
    }

    std::vector<FP> operator-(const std::vector<FP>& vec)
    {
        const size_t size = vec.size();
        const size_t end_of_vectorized_region = size - (size % 4);
        std::vector<FP> result(size);

#pragma omp parallel for
        for (size_t i = 0; i < end_of_vectorized_region; i += 4)
        {
            __m256d va = _mm256_loadu_pd(&vec[i]);
            __m256d vc = _mm256_set1_pd(-1);
            __m256d vr = _mm256_mul_pd(va, vc);
            _mm256_storeu_pd(&result[i], vr);
        }

        for (size_t i = end_of_vectorized_region; i < size; ++i)
            result[i] = vec[i] * -1;

        return result;
    }

    // Matrix interface
    class IMatrix
    {
    public:
        virtual ~IMatrix() = default;
        virtual FP at(size_t, size_t) = 0;
        virtual FP size() = 0; // ���������� �������
        virtual std::vector<FP> operator*(const std::vector<FP>&) = 0;

    };


    // Solver interface
    class ISolver
    {
    protected:
        std::vector<FP> initial_approximation;
        size_t max_iterations;
        FP required_precision;
        std::unique_ptr<IMatrix> system_matrix;
        std::vector<FP> b;

        ISolver() = default;

        ISolver(std::vector<FP> init_approx, size_t max_iters, FP required_precision, std::unique_ptr<IMatrix> system_matrix, std::vector<FP> b)
            : initial_approximation(std::move(init_approx)),
            max_iterations(max_iters),
            required_precision(required_precision),
            system_matrix(std::move(system_matrix)),
            b(b) {}


    public:
        virtual std::vector<FP> solve() const = 0;

        virtual ~ISolver() = default;

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

        void set_b(std::vector<FP> b)
        {
            this->b = std::move(b);
        }

    };


    // Minimal residual method
    class MinRes : public ISolver
    {

    public:
        MinRes(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix, std::vector<FP> b) :
            ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix), std::move(b)) {}

        MinRes() = default;
        ~MinRes() override = default;

        std::vector<FP> solve() const override
        {
            std::vector<FP> approximation = std::move(initial_approximation);
            std::vector<FP> residual;
            FP approximation_error, residual_norm;
            size_t i = 0;

            for (; i < max_iterations; ++i)
            {
                std::vector<FP> saved_approximation = approximation;

                residual = (*system_matrix) * approximation - b;

                std::vector<FP> Ar = (*system_matrix) * residual;
                FP tau = scalar_product(Ar, residual) / scalar_product(Ar, Ar);

                approximation = vector_FMA(residual, -tau, approximation);

                approximation_error = error(saved_approximation, approximation);
                if (approximation_error <= required_precision) break;
            }

            std::ofstream fout("../files/Solver_results.txt");
            fout << approximation_error << '\n' << norm(residual) << '\n' << i;
            fout.close();


            return approximation;
        }
    };

    class TopRelaxation: public ISolver
    {
        FP omega;
    public:
        TopRelaxation(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix, std::vector<FP> b, FP new_omega = 1.0):
        ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix), std::move(b)), omega(new_omega) {}

        TopRelaxation() = default;
        ~TopRelaxation() override = default;

        std::vector<FP> solve() const override
        {
            size_t size = initial_approximation.size();
            std::vector<FP> approximation = std::move(initial_approximation);
            size_t i = 0;
            FP approximation_error = 0;
            std::vector<FP> residual;
            
            //top relaxation main part
            for (; i < max_iterations; ++i)
            {
                std::vector<FP> saved_approximation = approximation;
                
                for(size_t ii = 0; ii < size; ++ii)
                {
                    FP approx_old = approximation[ii];
                    FP approx_new = (1 - omega) * system_matrix->at(ii, ii) * approximation[ii] + omega * b[ii];
                    
                    for(size_t j = 0; j < ii; ++j)
                        approx_new -= omega * system_matrix->at(ii, j) * approximation[j];

                    for(size_t j = ii + 1; j < size; ++j)
                        approx_new -= omega * system_matrix->at(ii, j) * approximation[j];
                    
                    approx_new /= system_matrix->at(ii, ii);
                    approximation[ii] = approx_new;
                }
                approximation_error = error(saved_approximation, approximation);
                if (approximation_error <= required_precision) break;
            }
            residual = (*system_matrix) * approximation - b; 

            std::ofstream fout("../files/Solver_results.txt");
            fout << approximation_error << '\n' << norm(residual) << '\n' << i;
            fout.close();

            return approximation;
        }
    };

    class TopRelaxationOptimizedForDirichletRegularGrid: public ISolver//without using grid
    {
        using one_dim_function = std::function<FP(FP)>;
        using two_dim_function = std::function<FP(FP, FP)>;

        two_dim_function f;
        one_dim_function mu1, mu2, mu3, mu4;

        std::array<FP, 4> corners;  // {x0, y0, xn, ym}

        FP omega;

        size_t n, m;

    public:
        TopRelaxationOptimizedForDirichletRegularGrid(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix, 
        std::vector<FP> b, two_dim_function new_f, one_dim_function new_mu1, one_dim_function new_mu2, one_dim_function new_mu3, one_dim_function new_mu4, 
        size_t new_n, size_t new_m, std::array<FP, 4> new_corners, FP new_omega = 1.0):
        ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix), std::move(b)), f(new_f), 
        mu1(new_mu1), mu2(new_mu2), mu3(new_mu3), mu4(new_mu4), n(new_n), m(new_m), corners(new_corners), omega(new_omega) {}

        TopRelaxationOptimizedForDirichletRegularGrid() = default;
        ~TopRelaxationOptimizedForDirichletRegularGrid() override = default;

        std::vector<FP> solve() const override
        {
            size_t size = initial_approximation.size();
            std::vector<FP> approximation = std::move(initial_approximation);
            std::vector<FP> residual; 
            FP residual_norm;

            std::vector<std::vector<FP>> approximation_matrix(n + 1, std::vector<FP>(m + 1));
            FP h = (corners[2] - corners[0]) / n;
            FP k = (corners[3] - corners[1]) / m;
            FP vertical_coef, horizonal_coef, A;
            horizonal_coef = -1.0 / (h * h);
            vertical_coef = -1.0 / (k * k);
            A = -2*(horizonal_coef + vertical_coef);
            size_t ind = 0;

            for(size_t j = 1; j < m; ++j)
                for(size_t i = 1; i < n; ++i)
                    approximation_matrix[i][j] = ind++;//approximation[ind++];

            for(size_t i = 0; i < n + 1; ++i)
            {
                approximation_matrix[i][0] = mu3(corners[0] + i * h);
                approximation_matrix[i][m] = mu4(corners[0] + i * h);
            }
            for(size_t j = 0; j < m + 1; ++j)
            {
                approximation_matrix[0][j] = mu1(corners[1] + j * k);
                approximation_matrix[n][j] = mu2(corners[1] + j * k);
            }

            size_t ii = 0;
            for (; ii < max_iterations; ++ii)
            {
                residual_norm = 0;
                for(size_t j = 1; j < m; ++j)
                    for(size_t i = 1; i < n; ++i)
                    {
                        FP x = corners[0] + i * h;
                        FP y = corners[1] + j * k;

                        FP approx_old = approximation_matrix[i][j];
                        FP approx_new = -omega * (horizonal_coef * (approximation_matrix[i + 1][j] + approximation_matrix[i - 1][j]) + vertical_coef * (approximation_matrix[i][j + 1] + approximation_matrix[i][j - 1]));
                        approx_new += (1 - omega) * A * approximation_matrix[i][j] + omega * f(x, y);
                        approx_new /= A;
                        approximation_matrix[i][j] = approx_new;

                        residual_norm = std::max(std::abs(approx_old - approx_new), residual_norm);
                    }
                if (residual_norm <= required_precision) break;
            }
            ind = 0;
            for(size_t i = 1; i < m; ++i)
                for(size_t j = 1; j < n; ++j)
                    approximation[ind++] = approximation_matrix[j][i];

            residual = (*system_matrix) * approximation - b; 

            std::ofstream fout("../files/Solver_results.txt");
            fout << residual_norm << '\n' << norm(residual) << '\n' << ii;
            fout.close();
            return approximation;
        }
    };


    // Chebyshev iteration method
    class ChebyshevIteration : public ISolver
    {
    private:
        FP Mmin;
        FP Mmax;

    public:
        ChebyshevIteration(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix, std::vector<FP> b, FP min = 0, FP max = 0) :
            ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix), std::move(b)), Mmin(min), Mmax(max) {}
        ChebyshevIteration() = default;
        ~ChebyshevIteration() = default;


        void set_Mmin(FP min) { Mmin = min; }
        void set_Mmax(FP max) { Mmax = max; }


        std::vector<FP> solve() const override
        {
            std::vector<FP> approximation = std::move(initial_approximation);
            double size = (*system_matrix).size();
            FP approximation_error = 0;
            size_t i = 0;

            FP k_cheb = 2.0;
            FP tau0 = 1.0 / ((Mmin + Mmax) / 2.0 + (Mmax - Mmin) / 2 * cos(PI / (2.0 * k_cheb) * (1.0 + 2.0 * 0.0)));
            FP tau1 = 1.0 / ((Mmin + Mmax) / 2.0 + (Mmax - Mmin) / 2 * cos(PI / (2.0 * k_cheb) * (1.0 + 2.0 * 1.0)));
            std::cout<<"Метод работает на основе оценок собсвтенных чисел"<<"\n";
            std::cout<<"k = 2"<<"\n";
            std::cout<<"tau1 = "<<tau0<<" tau2 = "<<tau1<<"\n";
            std::cout<<"Максимальное по модулю с.ч. "<<abs(Mmax)<<" Минимальное по модулю с.ч. "<<abs(Mmin)<<"\n";
            std::vector<FP> residual_first = (*system_matrix) * approximation - b;
            std::cout<<"Невязка на начальном приближении "<< norm(residual_first)<<"\n";
            for (; i < max_iterations; ++i)
            {
                std::vector<FP> saved_approximation = approximation;
                std::vector<FP> residual = (*system_matrix) * approximation - b;

                if (i % 2 == 0)
                    approximation = vector_FMA(residual, -tau0, approximation);
                else
                    approximation = vector_FMA(residual, -tau1, approximation);

                approximation_error = error(saved_approximation, approximation);
                if (approximation_error <= required_precision) break;
            }

            std::vector<FP> residual = (*system_matrix) * approximation - b;

            std::ofstream fout("../files/Solver_results.txt");
            fout << approximation_error << '\n' << norm(residual) << '\n' << i;
            fout.close();

            return approximation;
        }
    };


    // Conjugate gradient method
    class ConGrad : public ISolver
    {

    public:
        ConGrad(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix, std::vector<FP> b) :
            ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix), std::move(b)) {}

        ConGrad() = default;
        ~ConGrad() override = default;

        std::vector<FP> solve() const override
        {
            std::vector<FP> approximation = std::move(initial_approximation);
            FP approximation_error = 0;

            std::vector<FP> residual = (*system_matrix) * approximation - b;

            std::vector<FP> h = -residual;

            std::vector<FP> Ah = (*system_matrix) * h;

            FP alpha = -scalar_product(residual, h) / scalar_product(Ah, h);

            approximation = vector_FMA(h, alpha, approximation);
            size_t i = 1;

            for (; i < max_iterations; ++i)
            {
                std::vector<FP> saved_approximation = approximation;

                residual = (*system_matrix) * approximation - b;

                //FP residual_norm = norm(residual);

                FP beta = scalar_product(Ah, residual) / scalar_product(Ah, h);

                h = vector_FMA(h, beta, -residual);

                Ah = (*system_matrix) * h;

                alpha = -scalar_product(residual, h) / scalar_product(Ah, h);

                approximation = vector_FMA(h, alpha, approximation);

                approximation_error = error(saved_approximation, approximation);
                if (approximation_error <= required_precision) break;
            }

            residual = (*system_matrix) * approximation - b;

            std::ofstream fout("../files/Solver_results.txt");
            fout << approximation_error << '\n' << norm(residual) << '\n' << i;
            fout.close();

            return approximation;
        }
    };
    

} // namespace numcpp


#endif // __SOLVER_HPP__