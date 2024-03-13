#ifndef __SPLINE__
#define __SPLINE__

#include "root.h"
#include "TMA.h"
#include <functional>
#include <vector>
#include <iostream>
#include <array>

// class spline {
// private:
//     inline FP f_at(size_t i) const;
//     std::vector<FP> set_vector_c();
//     void set_vectors();

// public:
//     spline(FP l, FP r, size_t n, std::function<FP(FP)> f, FP mu1 = 0, FP mu2 = 0);
//     spline();
//     spline(std::function<FP(FP)> f, FP l, FP r, size_t n, FP mu1 = 0, FP mu2 = 0);
//     spline(std::pair<FP, FP> boundaries, size_t n, std::function<FP(FP)> f, std::pair<FP, FP> mu);
//     spline(std::function<FP(FP)> f, std::pair<FP, FP> boundaries, size_t n, std::pair<FP, FP> mu);

// public:
//     FP operator()(FP x);

// private:
//     FP l, r;
//     size_t n;
//     std::function<FP(FP)> f;
//     FP mu1, mu2;
//     FP h;

//     std::vector<FP> a, b, c, d;
// };

class spline {
private:
    // Calculates f_i
    // Implementation may be changed due to 0-indexing. Do it if you need.
    inline FP f_at(size_t i) const {
        return f(l + i * h);
    }

    void set_vectors() {
        a = b = c = d = std::vector<FP>(n + 1, 0.0);
        
        // Here set other vectors
        for (size_t i = 0; i <= n; ++i) {
            a[i] = f_at(i);
        }

        TMA run_through(n, a, h, std::make_pair(mu1, mu2));
        c = run_through.get_solution();

        for (size_t i = 1; i <= n; ++i) {
            d[i] = (c[i] - c[i - 1]) / h;
            b[i] = (a[i] - a[i - 1]) / h + c[i] * h / 3.0 + c[i - 1] * h / 6.0;
        }
    }

public:
    spline(FP l, FP r, size_t n, std::function<FP(FP)> f, FP mu1 = 0, FP mu2 = 0): l(l), r(r), n(n), f(f), mu1(mu1), mu2(mu2) {
        h = (r - l) / static_cast<FP>(n);
        set_vectors();
    }

    spline() = default;
    // Constructors with different order of arguments. All delegates to basic one
    spline(std::function<FP(FP)> f, FP l, FP r, size_t n, FP mu1 = 0, FP mu2 = 0): spline(l, r, n, f, mu1, mu2) {}
    spline(std::pair<FP, FP> boundaries, size_t n, std::function<FP(FP)> f, std::pair<FP, FP> mu): spline(boundaries.first, boundaries.second, n, f, mu.first, mu.second) {}
    spline(std::function<FP(FP)> f, std::pair<FP, FP> boundaries, size_t n, std::pair<FP, FP> mu): spline(boundaries.first, boundaries.second, n, f, mu.first, mu.second) {}

public:
    // Main function of this class. Return appoximated value at x
    FP operator()(FP x) {
        return 3.14;
    }
    void show_vectors() {
        for (FP el : a) {
            std::cout << el << ", ";
        }
        std::cout << std::endl;
        for (FP el : b) {
            std::cout << el << ", ";
        }
        std::cout << std::endl;
        for (FP el : c) {
            std::cout << el << ", ";
        }
        std::cout << std::endl;
        for (FP el : d) {
            std::cout << el << ", ";
        }
        std::cout << std::endl;
    }

private:
    FP l, r;                        // Range
    size_t n;
    std::function<FP(FP)> f;
    FP mu1, mu2;                    // Values of second derivative
    FP h;
    std::vector<FP> a, b, c, d;     // Vectors of coeffs for polynoms
    
};

#endif // __SPLINE__