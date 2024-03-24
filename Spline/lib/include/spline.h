#ifndef __SPLINE__
#define __SPLINE__

#include "root.h"
#include "TMA.h"
#include <functional>
#include <vector>
#include <iostream>
#include <array>
#include <cmath>

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
    spline(FP l, FP r, size_t n, std::function<FP(FP)> f, std::function<FP(FP)> f1, std::function<FP(FP)> f2, FP mu1 = 0, FP mu2 = 0): 
        l(l), r(r), n(n), f(f), f1(f1), f2(f2), mu1(mu1), mu2(mu2) {
        h = (r - l) / static_cast<FP>(n);
        set_vectors();
    }

    spline() = default;

public:
    // Main function of this class. Return appoximated value at x
    FP operator()(FP x) {
        FP x_left, x_right;
        for (int i = 0; i < n; i++) {
            x_left = l + (FP)(i) * h;
            x_right = l + (FP)(i + 1) * h;
            if (x >= x_left && x <= x_right)
                return a[i + 1] + b[i + 1] * (x - x_right) + c[i + 1] / 2.0 * pow((x - x_right), 2.0) + d[i + 1] / 6.0 * pow((x - x_right), 3.0);
        }
        return 0.0;
    }
    FP get_a(int i) {
        return a[i];
    }
    FP get_b(int i) {
        return b[i];
    }
    FP get_c(int i) {
        return c[i];
    }
    FP get_d(int i) {
        return d[i];
    }

    FP get_f(FP i){
        return f(i);
    } 

    FP get_f1(FP i){
        return f1(i);
    }

    FP get_f2(FP i){
        return f2(i);
    }

    FP get_s1(FP x){
        FP x_left, x_right;
        for (size_t i = 0; i < n; i++) {
            x_left = l + static_cast<FP>(i) * h;
            x_right = l + static_cast<FP>(i + 1) * h;
            if (x >= x_left && x <= x_right)
                return (b[i + 1] + c[i + 1] * (x - x_right) + d[i + 1] / 2.0 * pow((x - x_right), 2.0));
        }
        return 0.0;
    }

    FP get_s2(FP x){
        FP x_left, x_right;
        for (size_t i = 0; i < n; i++) {
            x_left = l + static_cast<FP>(i) * h;
            x_right = l + static_cast<FP>(i + 1) * h;
            if (x >= x_left && x <= x_right)
                return (c[i + 1] + d[i + 1] * (x - x_right));
        }
        return 0.0;
    }

    FP error(FP x) {
        FP x_left, x_right;
        for (size_t i = 0; i < n; i++) {
            x_left = l + static_cast<FP>(i) * h;
            x_right = l + static_cast<FP>(i + 1) * h;
            if (x >= x_left && x <= x_right)
                return fabs(f(x) - (a[i + 1] + b[i + 1] * (x - x_right) + c[i + 1] / 2.0 * pow((x - x_right), 2.0) + d[i + 1] / 6.0 * pow((x - x_right), 3.0)));
        }
        return 0.0;
    }
    FP derivative_error(FP x) {
        FP x_left, x_right;
        for (size_t i = 0; i < n; i++) {
            x_left = l + static_cast<FP>(i) * h;
            x_right = l + static_cast<FP>(i + 1) * h;
            if (x >= x_left && x <= x_right)
                return fabs(f1(x) - (b[i + 1] + c[i + 1] * (x - x_right) + d[i + 1] / 2.0 * pow((x - x_right), 2.0)));
        }
        return 0.0;
    }
    FP second_derivative_error(FP x) {
        FP x_left, x_right;
        for (size_t i = 0; i < n; i++) {
            x_left = l + static_cast<FP>(i) * h;
            x_right = l + static_cast<FP>(i + 1) * h;
            if (x >= x_left && x <= x_right)
                return fabs(f2(x) - (c[i + 1] + d[i + 1] * (x - x_right)));
        }
        return 0.0;
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
    std::function<FP(FP)> f1;
    std::function<FP(FP)> f2;
    FP mu1, mu2;                    // Values of second derivative
    FP h;
    std::vector<FP> a, b, c, d;     // Vectors of coeffs for polynoms
    
};

#endif // __SPLINE__