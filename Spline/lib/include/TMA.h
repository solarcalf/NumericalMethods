#ifndef __TMA__
#define __TMA__

#include "root.h"

class TMA {
private:
	std::vector<FP> v;    // Numerical solution
	FP h;                 // Space step
	FP Ai, Bi, Ci;        // Coefficients
	size_t n;             // Grid size

	void run(std::pair<FP, FP> boundaries, std::vector<FP> f);

public:
	TMA(size_t size, std::vector<FP> source_function, FP step,
		std::pair<FP, FP> boundary_condtitions) : n(size) {
		v = std::vector<FP>(n + 1, 0.0);

		h = step;
		Ai = Bi = h;
		Ci = -4.0 * h;

		run(boundary_condtitions, source_function);
	}

	std::vector<FP> get_solution() { return v; }
};

void TMA::run(std::pair<FP, FP> boundaries, std::vector<FP> f) {
	FP mu1 = boundaries.first;
	FP mu2 = boundaries.second;
	std::vector<FP> alpha;
	std::vector<FP> beta;
	FP phi = 0.0;

	// Straight running
	beta.push_back(mu1);
	alpha.push_back(0.0);
	for (size_t i = 1; i < n; i++) {
		alpha.push_back(Bi / (Ci - Ai * alpha[i-1]));
		phi = -6.0 * ((f[i + 1] - f[i]) / h - (f[i] - f[i - 1]) / h);
		beta.push_back((phi + Ai * beta[i-1]) / (Ci - alpha[i - 1] * Ai));
	}

	// Reverse running
	v[n] = mu2;
	for (size_t i = n - 1; i > 0; i--) {
		v[i] = alpha[i] * v[i] + beta[i];
	}
	v[0] = mu1;
}

#endif // __TMA__
