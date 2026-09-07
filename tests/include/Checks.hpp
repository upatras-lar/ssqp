#ifndef SSQP_TESTS_CHECKS_HPP
#define SSQP_TESTS_CHECKS_HPP

#include "types.hpp"
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ssqp
{
namespace tests
{
inline void require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

// Componentwise tolerance prevents a large derivative from hiding errors in
// smaller components. Reject nonfinite values explicitly (NaN > tol is false).
template<class A, class B>
bool close(const Eigen::MatrixBase<A> &actual,
           const Eigen::MatrixBase<B> &expected, double atol = 2e-6,
           double rtol = 2e-6)
{
    return actual.rows() == expected.rows() &&
           actual.cols() == expected.cols() && actual.allFinite() &&
           expected.allFinite() &&
           ((actual - expected).array().abs() <=
            atol + rtol * expected.array().abs())
             .all();
}

template<class A, class B>
void requireClose(const Eigen::MatrixBase<A> &actual,
                  const Eigen::MatrixBase<B> &expected,
                  const std::string &context, double atol = 2e-6,
                  double rtol = 2e-6)
{
    if (close(actual, expected, atol, rtol)) return;
    std::ostringstream log;
    log << std::setprecision(17) << context << " (atol=" << atol
        << ", rtol=" << rtol << ")\nActual:\n"
        << actual << "\nExpected:\n"
        << expected;
    throw std::runtime_error(log.str());
}

// Refine using only function values, independently of the analytic gradient.
// abs/max penalties are piecewise smooth even after log-sum-exp smoothing:
// shrink the stencil until it stays within one differentiable branch.
template<class F>
double finiteDifference(F evaluate, double h)
{
    double coarse = (evaluate(h) - evaluate(-h)) / (2 * h);
    for (int refinement = 0; refinement < 10; ++refinement)
    {
        h           *= 0.5;
        double fine  = (evaluate(h) - evaluate(-h)) / (2 * h);
        if (std::abs(fine - coarse) <= 2e-7 * (1.0 + std::abs(fine)))
            return fine + (fine - coarse) / 3.0;
        coarse = fine;
    }
    // The comparison still fails if this estimate is nonfinite or inaccurate.
    return coarse;
}
}
}
#endif
