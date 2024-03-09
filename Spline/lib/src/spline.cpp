#include "../include/spline.h"


// class spline {
// private:
//     // Calculates f_i
//     // Implementation may be changed due to 0-indexing. Do it if you need.
//     inline FP f_at(size_t i) const {
//         return f(l + i * h);
//     }

//     std::vector<FP> set_vector_c() {
//         return {0}; // Placeholder
//     }

//     void set_vectors() {
//         std::vector<double> a(n), b(n), c(n), d(n);

//         c = set_vector_c();

//         // Here set other vectors
//         for (size_t i = 0; i < n; ++ i) {
//             // a[i] = ...
//             // b[i] = ...
//             // d[i] = ...
//         }
//     }

// public:
//     spline(FP l, FP r, size_t n, std::function<FP(FP)> f, FP mu1 = 0, FP mu2 = 0): l(l), r(r), n(n), f(f), mu1(mu1), mu2(mu2) {
//         FP h = (r - l) / static_cast<FP>(n);
//         set_vectors();
//     }

//     spline() = default;
//     // Constructors with different order of arguments. All delegates to basic one
//     spline(std::function<FP(FP)> f, FP l, FP r, size_t n, FP mu1 = 0, FP mu2 = 0): spline(l, r, n, f, mu1, mu2) {}
//     spline(std::pair<FP, FP> boundaries, size_t n, std::function<FP(FP)> f, std::pair<FP, FP> mu): spline(boundaries.first, boundaries.second, n, f, mu.first, mu.second) {}
//     spline(std::function<FP(FP)> f, std::pair<FP, FP> boundaries, size_t n, std::pair<FP, FP> mu): spline(boundaries.first, boundaries.second, n, f, mu.first, mu.second) {}

// public:
//     // Main function of this class. Return appoximated value at x
//     FP operator()(FP x) {
//         return 3.14;
//     }

// private:
//     FP l, r;    // Range
//     size_t n;
//     std::function<FP(FP)> f;
//     FP mu1, mu2;    // Values of second derivative
//     FP h;

//     std::vector<FP> a, b, c, d;     // Vectors of coeffs for polynoms
// };