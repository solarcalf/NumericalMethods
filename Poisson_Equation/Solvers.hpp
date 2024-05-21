#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#include <vector>
#include <algorithm>
#include <memory>
#include <omp.h>
#include <immintrin.h>
#include <cmath>

using FP = double;

namespace numcpp 
{

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

// Matrix interface
class IMatrix 
{
public: 
    virtual ~IMatrix() = default;
    virtual FP at(size_t, size_t) = 0;
    virtual FP size() = 0; // квадратная матрица
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
        this-> b = std::move(b);
    }

};


// Minimal residual method
class MinRes: public ISolver 
{
public:
    MinRes(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix, std::vector<FP> b):
    ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix), std::move(b)) {}

    MinRes() = default;
    ~MinRes() override = default;

    std::vector<FP> solve() const override
    {
        std::vector<FP> approximation = std::move(initial_approximation);

        for (size_t i = 0; i < max_iterations; ++i)
        {
            std::vector<FP> residual = (*system_matrix) * approximation - b;
            
            FP residual_norm = norm(residual);
            if (residual_norm <= required_precision) break;

            std::vector<FP> Ar = (*system_matrix) * residual;
            FP tau = scalar_product(Ar, residual) / scalar_product(Ar, Ar);

            approximation = vector_FMA(residual, -tau, approximation);
        }

        return approximation;
    }
};


 // Chebyshev iteration method
class ChebyshevIteration: public ISolver
{
    private:
        FP Mmin;
        FP Mmax;
    public:
        ChebyshevIteration(std::vector<FP> initial_approximation, size_t max_iterations, FP required_precision, std::unique_ptr<IMatrix> system_matrix, std::vector<FP> b):
        ISolver(std::move(initial_approximation), max_iterations, required_precision, std::move(system_matrix), std::move(b)) {}

        ChebyshevIteration() = default;
        ~ChebyshevIteration() = default;


        void set_Mmin(FP min){ Mmin = min; }
        void set_Mmax(FP max){ Mmax = max; }


        std::vector<FP> solve() const override
        {
            std::vector<FP> approximation = std::move(initial_approximation);
            double size = (*system_matrix).size();
            //FP h = sqrt(1.0 / (*system_matrix).at(0, 1));
            //FP k = sqrt(1.0 / (*system_matrix).at(0, sqrt(size)));

            FP k_cheb = 2.0;
            FP tau0 = 1.0 / ((Mmin + Mmax) / 2.0 + (Mmax - Mmin) / 2 * cos(PI / (2.0 * k_cheb) * (1.0 + 2.0 * 0.0)));
            FP tau1 = 1.0 / ((Mmin + Mmax) / 2.0 + (Mmax - Mmin) / 2 * cos(PI / (2.0 * k_cheb) * (1.0 + 2.0 * 1.0)));


            for (size_t i = 0; i < max_iterations; ++i)
            {
                std::vector<FP> residual = (*system_matrix) * approximation - b;

                FP residual_norm = norm(residual);
                if (residual_norm <= required_precision) break;

                if (i % 2 == 0)
                    approximation = vector_FMA(residual, -tau0, approximation);
                else
                    approximation = vector_FMA(residual, -tau1, approximation);
            }

            return approximation;
        }
        }
};
    

} // namespace numcpp


#endif // __SOLVER_HPP__
