#include "../include/spline.h"

static FP l, r;
static size_t function_num;
static spline s;

std::array<const std::function<FP(FP)>, 4> functions {
    [](FP x){return x;}, 
    [](FP x){return x;}, 
    [](FP x){return x;}, 
    [](FP x){return x;}
};

extern "C" void set_spline(FP l, FP r, uint32_t n, uint8_t fun_num, FP mu1 = 0, FP mu2 = 0) {
    s = spline(l, r, n, functions[fun_num], mu1, mu2);
}

extern "C" FP* get_approximation(FP a, FP b, uint32_t n) {
    FP* approximation = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 0; i < n; ++i)
        approximation[i] = s(a + i*h);

    return approximation; // Memory leak
}