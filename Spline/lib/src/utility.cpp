#include "../include/spline.h"

static FP l, r;
static size_t function_num; 
static spline s;
static FP* approximation;
static FP* f;
static FP* f1;
static FP* f2;
static FP* s1;
static FP* s2;
static FP* ai;
static FP* bi;
static FP* ci;
static FP* di;
static FP* error;
static FP* derivative_error;
static FP* second_derivative_error;


std::array<const std::function<FP(FP)>, 5> functions {
    [](FP x) {
        if (-1.0 <= x && x < 0.0)
            return pow(x, 3.0) + 3.0 * pow(x, 2.0);
        if (0.0 <= x && x <= 1.0)
            return -pow(x, 3.0) + 3.0 * pow(x, 2.0);
    },
    [](FP x) {return sin(exp(x)); }, 
    [](FP x) {return sin(cos(x)); },
    [](FP x) {return sin(x) / x; },
    [](FP x) {return exp(x - 3); }
};

std::array<const std::function<FP(FP)>, 5> derivatives{
    [](FP x) {
        if (-1.0 <= x && x < 0.0)
            return 3.0 * pow(x, 2.0) + 6.0 * x;
        if (0.0 <= x && x <= 1.0)
            return -3.0 * pow(x, 2.0) + 6.0 * x;
    },
    [](FP x) {return exp(x) * cos(exp(x)); },
    [](FP x) {return -sin(x) * cos(cos(x)); },
    [](FP x) {return -(sin(x) - x * cos(x)) / (x * x); },
    [](FP x) {return exp(x - 3); }
};

std::array<const std::function<FP(FP)>, 5> second_derivatives{
    [](FP x) {
        if (-1.0 <= x && x < 0.0)
            return 6.0 * x + 6.0;
        if (0.0 <= x && x <= 1.0)
            return -6.0 * x + 6.0;
    },
    [](FP x) {return exp(x) * cos(exp(x)) - exp(2.0 * x) * sin(exp(x)); },
    [](FP x) {return -pow(sin(x), 2.0) * sin(cos(x)) - cos(x) * cos(cos(x)); },
    [](FP x) {return -((x * x - 2) * sin(x) + 2.0 * x * cos(x)) / (x * x * x); },
    [](FP x) {return exp(x - 3); }
};

extern "C" void set_spline(FP l, FP r, uint32_t n, uint8_t fun_num, FP mu1 = 0, FP mu2 = 0) {
    s = spline(l, r, n, functions[fun_num], derivatives[fun_num], second_derivatives[fun_num], mu1, mu2);
}

extern "C" FP* get_approximation(FP a, FP b, uint32_t n) {
    approximation = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 0; i < n; ++i)
        approximation[i] = s(a + i*h);

    return approximation;
}

extern "C" FP* get_f(FP a, FP b, uint32_t n) {
    f = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 0; i < n; ++i)
        f[i] = s.get_f(a + i*h);

    return f;
}

extern "C" FP* get_f1(FP a, FP b, uint32_t n) {
    f1 = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 0; i < n; ++i)
        f1[i] = s.get_f1(a + i*h);

    return f1;
}

extern "C" FP* get_f2(FP a, FP b, uint32_t n) {
    f2 = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 0; i < n; ++i)
        f2[i] = s.get_f2(a + i*h);

    return f2;
}

extern "C" FP* get_s1(FP a, FP b, uint32_t n) {
    s1 = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 0; i < n; ++i)
        s1[i] = s.get_s1(a + i*h);

    return s1;
}

extern "C" FP* get_s2(FP a, FP b, uint32_t n) {
    s2 = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 0; i < n; ++i)
        s2[i] = s.get_s2(a + i*h);

    return s2;
}


extern "C" FP * get_a(uint32_t n)
{
    ai = (FP*)malloc(n * sizeof(FP));

    for (size_t i = 1; i <= n; ++i)
        ai[i - 1] = s.get_a(i);

    return ai;
}

extern "C" FP * get_b(uint32_t n)
{
    bi = (FP*)malloc(n * sizeof(FP));

    for (size_t i = 1; i <= n; ++i)
        bi[i - 1] = s.get_b(i);

    return bi;
}

extern "C" FP * get_c(uint32_t n)
{
    ci = (FP*)malloc(n * sizeof(FP));

    for (size_t i = 1; i <= n; ++i)
        ci[i - 1] = s.get_c(i);

    return ci;
}

extern "C" FP * get_d( uint32_t n)
{
    di = (FP*)malloc(n * sizeof(FP));

    for (size_t i = 1; i <= n; ++i)
        di[i - 1] = s.get_d(i);

    return di;
}

extern "C" FP* get_error(FP a, FP b, uint32_t n) {
    error = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 1; i <= n; ++i)
        error[i - 1] = s.error(a + i * h);

    return error;
}

extern "C" FP* get_derivative_error(FP a, FP b, uint32_t n) {
    derivative_error = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 1; i <= n; ++i)
        derivative_error[i - 1] = s.derivative_error(a + i * h);

    return derivative_error;
}

extern "C" FP* get_second_derivative_error(FP a, FP b, uint32_t n) {
    second_derivative_error = (FP*)malloc(n * sizeof(FP));
    FP h = (b - a) / static_cast<FP>(n);

    for (size_t i = 1; i <= n; ++i)
        second_derivative_error[i - 1] = s.second_derivative_error(a + i * h);

    return second_derivative_error;
}

extern "C" void free_all() {
    free(approximation);
    free(f);
    free(ai);
    free(bi);
    free(ci);
    free(di);
    free(error);
    free(derivative_error);
    free(second_derivative_error);
}

extern "C" void free_some(uint8_t i) {
    if (i == 0){
        free(approximation);
        free(f);
        free(error);
    }else if (i == 1){
        free(f1);
        free(s1);
        free(derivative_error);
    }else{
        free(f2);
        free(s2);
        free(second_derivative_error);
    }

    

}
