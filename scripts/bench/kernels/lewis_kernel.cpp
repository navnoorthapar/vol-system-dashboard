// lewis_kernel.cpp: the Heston Lewis-contour sum on a FIXED trapezoid grid, in C++,
// for the benchmark in scripts/bench/test_bench_kernels.py only. Nothing in the
// package calls it. It evaluates exactly what models/heston.py::_lewis_call_prices
// does once the grid is known (the "little trap" CF on u - i/2, the Black-Scholes
// control variate, the trapezoid weights) so its prices can be checked against the
// NumPy engine before it is timed.
//
//   direct:     acc_j = sum_k Re(exp(i u_k x_j) gw_k)      one sincos per (k, j)
//   recurrence: the phase exp(i u_k x_j) is advanced by one complex multiply per
//               (k, j) (u_k = k h, so exp(i u_{k+1} x) = exp(i u_k x) exp(i h x))
//
// Build: g++ -O3 -std=c++17 -shared -fPIC lewis_kernel.cpp -o liblewis.so
// (no -ffast-math: IEEE semantics, as NumPy and Numba use by default).
#include <cmath>
#include <complex>

using cd = std::complex<double>;

static inline cd clog1p(cd z) {
    // log(1 + z), accurate for small |z| (matches heston._clog1p)
    const double re = z.real(), im = z.imag();
    return cd(0.5 * std::log1p(2.0 * re + re * re + im * im), std::atan2(im, 1.0 + re));
}

static inline cd cexpm1(cd z) {
    // exp(z) - 1 without cancellation for small |z|
    if (std::abs(z) < 1e-5) return z + 0.5 * z * z + z * z * z / 6.0;
    return std::exp(z) - 1.0;
}

static inline cd heston_log_cf(cd z, double T, double kappa, double theta, double sigma,
                               double rho, double v0) {
    const double sig = sigma > 1e-12 ? sigma : 1e-12;
    const cd I(0.0, 1.0);
    const cd s = z * (z + I);
    const cd beta = kappa - I * rho * sig * z;
    const cd d = std::sqrt(beta * beta + sig * sig * s);
    const cd bpd = beta + d;
    const cd one_m_e = -cexpm1(-d * T);
    const cd e = 1.0 - one_m_e;
    const cd g = -sig * sig * s / (bpd * bpd);
    const cd D = -s * one_m_e / (bpd * (1.0 - g * e));
    const cd log_ratio = clog1p(g * one_m_e / (1.0 - g));
    const cd C = kappa * theta * (-s * T / bpd - 2.0 * log_ratio / (sig * sig));
    return C + D * v0;
}

// gw_k = h * (exp(-w (u^2 + 1/4) / 2) - phi(u - i/2)) / (u^2 + 1/4), halved at k = 0
static void weights(int n, double h, double T, double kappa, double theta, double sigma,
                    double rho, double v0, double w, cd* gw) {
    for (int k = 0; k < n; ++k) {
        const double u = h * k;
        const double uu = u * u + 0.25;
        const cd phi = std::exp(heston_log_cf(cd(u, -0.5), T, kappa, theta, sigma, rho, v0));
        gw[k] = (std::exp(-0.5 * w * uu) - phi) / uu * h;
    }
    gw[0] *= 0.5;
}

extern "C" {

// acc[j] = sum_k Re(exp(i u_k x_j) gw_k), j < m; recurrence != 0 selects the phase recurrence
void lewis_sum(int n, double h, double T, double kappa, double theta, double sigma, double rho,
               double v0, double w, int m, const double* x, int recurrence, double* acc) {
    cd* gw = new cd[n];
    weights(n, h, T, kappa, theta, sigma, rho, v0, w, gw);
    for (int j = 0; j < m; ++j) {
        double a = 0.0;
        if (recurrence) {
            const cd rot = std::exp(cd(0.0, h * x[j]));
            cd cur(1.0, 0.0);
            for (int k = 0; k < n; ++k) {
                a += cur.real() * gw[k].real() - cur.imag() * gw[k].imag();
                cur *= rot;
            }
        } else {
            for (int k = 0; k < n; ++k) {
                const double ph = h * k * x[j];
                a += std::cos(ph) * gw[k].real() - std::sin(ph) * gw[k].imag();
            }
        }
        acc[j] = a;
    }
    delete[] gw;
}

}  // extern "C"
