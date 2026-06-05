#include "types.hpp"
#include "model/Model.hpp"
#include "model/stagewise/StageDynamics.hpp"

#include <Eigen/Dense>

namespace ssqp
{
namespace model
{
namespace custom
{
namespace double_pendulum
{
/* Custom stagewise dynamics class for the double pendulum system
 * Defines the double pendulum dynamics relative to 6 parameters:
 *  1. Mass 1 (m1)
 *  2. Mass 2 (m2)
 *  3. Length 1 (l1)
 *  4. Length 2 (l2)
 *  5. Inertia 1 (I1)
 *  6. Inertia 2 (I2)
 */
class DoublePendulumDynamics : public stagewise::StageDynamics
{
  public:
    DoublePendulumDynamics(const Vec6 &params, const double dt = 0.01,
                           const std::string &name = "")
      : StageDynamics(name)
      , _params(params)
      , _dt(dt)
    {
    }

    uint output_size() const override
    {
        return _nq;
    }
    uint state_size() const override
    {
        return _nq;
    }
    uint control_size() const override
    {
        return _nu;
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        // States & Controls
        const double q1  = x(0);
        const double q2  = x(1);
        const double dq1 = x(2);
        const double dq2 = x(3);

        const double tau1 = u(0);
        const double tau2 = u(1);

        // Parameter aliases
        const double m1 = _params[0];
        const double m2 = _params[1];
        const double l1 = _params[2];
        const double l2 = _params[3];
        const double I1 = _params[4];
        const double I2 = _params[5];
        const double g  = 9.81;

        // Center of mass lengths (uniform mass)
        const double lc1 = 0.5 * l1;
        const double lc2 = 0.5 * l2;

        // Precompute trigonometric functions
        double s2  = std::sin(q2);
        double c2  = std::cos(q2);
        double s1  = std::sin(q1);
        double s12 = std::sin(q1 + q2);

        // Mass Matrix elements: M = [M11, M12; M21, M22]
        double M11 = I1 + I2 + m1 * lc1 * lc1 +
                     m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * c2);
        double M12 = I2 + m2 * (lc2 * lc2 + l1 * lc2 * c2);
        double M21 = M12;
        double M22 = I2 + m2 * lc2 * lc2;

        // Coriolis, Centrifugal, and Gravity elements: V = C(q, dq)*dq + G(q)
        double V1 = -m2 * l1 * lc2 * dq2 * (2.0 * dq1 + dq2) * s2 +
                    (m1 * lc1 + m2 * l1) * g * s1 + m2 * g * lc2 * s12;

        double V2 = m2 * l1 * lc2 * dq1 * dq1 * s2 + m2 * g * lc2 * s12;

        // Generalized forces acting on the joints
        double f1 = tau1 - V1;
        double f2 = tau2 - V2;

        // Solve for accelerations (M * ddq = f) using 2x2 matrix determinant inversion
        double det = M11 * M22 - M12 * M21;

        // Guard against numerical issues (determinant of a physical mass matrix shouldn't be 0)
        if (std::abs(det) < 1e-9)
        {
            det = (det >= 0) ? 1e-9 : -1e-9;
        }

        double ddq1 = (M22 * f1 - M12 * f2) / det;
        double ddq2 = (-M21 * f1 + M11 * f2) / det;

        // Semi-Implicit Euler Integration Steps
        // 1. Update velocities first using current accelerations
        double next_dq1 = dq1 + ddq1 * _dt;
        double next_dq2 = dq2 + ddq2 * _dt;

        // 2. Update positions using the newly computed next velocities
        double next_q1 = q1 + next_dq1 * _dt;
        double next_q2 = q2 + next_dq2 * _dt;

        // Pack into the next state vector
        VecX next_x(_nq);
        next_x << next_q1, next_q2, next_dq1, next_dq2;

        return next_x;
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        // Resize output matrices
        dX.setZero(_nq, _nq);
        dU.setZero(_nq, _nu);

        // States & Controls
        double q1  = x(0);
        double q2  = x(1);
        double dq1 = x(2);
        double dq2 = x(3);

        double tau1 = u(0);
        double tau2 = u(1);

        // Parameter aliases
        const double m1 = _params[0];
        const double m2 = _params[1];
        const double l1 = _params[2];
        const double l2 = _params[3];
        const double I1 = _params[4];
        const double I2 = _params[5];
        const double g  = 9.81;

        double lc1 = 0.5 * l1;
        double lc2 = 0.5 * l2;

        // Trigonometric functions
        double s2  = std::sin(q2);
        double c2  = std::cos(q2);
        double s1  = std::sin(q1);
        double c1  = std::cos(q1);
        double s12 = std::sin(q1 + q2);
        double c12 = std::cos(q1 + q2);

        // --- Forward Dynamics (re-computed to get current accelerations ddq) ---
        Mat2 M;
        M(0, 0) = I1 + I2 + m1 * lc1 * lc1 +
                  m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * c2);
        M(0, 1) = I2 + m2 * (lc2 * lc2 + l1 * lc2 * c2);
        M(1, 0) = M(0, 1);
        M(1, 1) = I2 + m2 * lc2 * lc2;

        double V1 = -m2 * l1 * lc2 * dq2 * (2.0 * dq1 + dq2) * s2 +
                    (m1 * lc1 + m2 * l1) * g * s1 + m2 * g * lc2 * s12;

        double V2 = m2 * l1 * lc2 * dq1 * dq1 * s2 + m2 * g * lc2 * s12;

        Vec2 f(tau1 - V1, tau2 - V2);
        Mat2 M_inv = M.inverse();
        Vec2 ddq   = M_inv * f;

        // --- Analytical Partial Derivatives of M and V ---

        // 1. Partial derivatives with respect to q1
        // M does not depend on q1, so dM_dq1 = 0
        Vec2 dV_dq1((m1 * lc1 + m2 * l1) * g * c1 + m2 * g * lc2 * c12,
                    m2 * g * lc2 * c12);

        // 2. Partial derivatives with respect to q2
        Mat2 dM_dq2;
        dM_dq2(0, 0) = -2.0 * m2 * l1 * lc2 * s2;
        dM_dq2(0, 1) = -m2 * l1 * lc2 * s2;
        dM_dq2(1, 0) = dM_dq2(0, 1);
        dM_dq2(1, 1) = 0.0;

        Vec2 dV_dq2(-m2 * l1 * lc2 * dq2 * (2.0 * dq1 + dq2) * c2 +
                      m2 * g * lc2 * c12,
                    m2 * l1 * lc2 * dq1 * dq1 * c2 + m2 * g * lc2 * c12);

        // 3. Partial derivatives with respect to dq1
        Vec2 dV_ddq1(-2.0 * m2 * l1 * lc2 * dq2 * s2,
                     2.0 * m2 * l1 * lc2 * dq1 * s2);

        // 4. Partial derivatives with respect to dq2
        Vec2 dV_ddq2(-2.0 * m2 * l1 * lc2 * (dq1 + dq2) * s2, 0.0);

        // --- Formulate partials of acceleration: ddq_dq and ddq_ddq ---
        Mat2 ddq_dq;
        // Column 0: partials w.r.t q1
        ddq_dq.col(0) = M_inv * (-dV_dq1);
        // Column 1: partials w.r.t q2 (includes dM/dq2 term)
        ddq_dq.col(1) = M_inv * (-dV_dq2 - dM_dq2 * ddq);

        Mat2 ddq_ddq;
        // Column 0: partials w.r.t dq1, Column 1: partials w.r.t dq2
        ddq_ddq.col(0) = M_inv * (-dV_ddq1);
        ddq_ddq.col(1) = M_inv * (-dV_ddq2);

        // Partials w.r.t control u
        Mat2 ddq_du = M_inv;

        // --- Construct the final dX and dU block matrices using Semi-Implicit Euler relations ---

        // Identity matrix block helpers
        Mat2 I = Mat2::Identity();

        // Upper rows of dX (partials of next_q)
        dX.block<2, 2>(0, 0) = I + ddq_dq * (_dt * _dt);
        dX.block<2, 2>(0, 2) = I * _dt + ddq_ddq * (_dt * _dt);

        // Lower rows of dX (partials of next_dq)
        dX.block<2, 2>(2, 0) = ddq_dq * _dt;
        dX.block<2, 2>(2, 2) = I + ddq_ddq * _dt;

        // Control Jacobian dU
        dU.block<2, 2>(0, 0) = ddq_du * (_dt * _dt);
        dU.block<2, 2>(2, 0) = ddq_du * _dt;
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxx) const override
    {
        Pxx.setZero(_nq, _nq);
    }
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxu) const override
    {
        Pxu.setZero(_nq, _nu);
    }
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Puu) const override
    {
        Puu.setZero(_nu, _nu);
    }

    uint parameter_size() const override
    {
        return 6;
    }

    VecX parameters() const override
    {
        return _params;
    }

    // TODO: verify
    MatX parameter_gradient(const VecX &x, const VecX &u) const override
    {
        // States & Controls
        const double q1  = x(0);
        const double q2  = x(1);
        const double dq1 = x(2);
        const double dq2 = x(3);

        const double tau1 = u(0);
        const double tau2 = u(1);

        // Parameter aliases
        const double m1 = _params[0];
        const double m2 = _params[1];
        const double l1 = _params[2];
        const double l2 = _params[3];
        const double I1 = _params[4];
        const double I2 = _params[5];
        const double g  = 9.81;

        // Center of mass lengths (uniform mass)
        const double lc1 = 0.5 * l1;
        const double lc2 = 0.5 * l2;

        // Precompute trigonometric functions
        double s2  = std::sin(q2);
        double c2  = std::cos(q2);
        double s1  = std::sin(q1);
        double s12 = std::sin(q1 + q2);

        // Mass Matrix elements: M = [M11, M12; M21, M22]
        double M11 = I1 + I2 + m1 * lc1 * lc1 +
                     m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * c2);
        double M12 = I2 + m2 * (lc2 * lc2 + l1 * lc2 * c2);
        double M21 = M12;
        double M22 = I2 + m2 * lc2 * lc2;

        // Coriolis, Centrifugal, and Gravity elements: V = C(q, dq)*dq + G(q)
        double V1 = -m2 * l1 * lc2 * dq2 * (2.0 * dq1 + dq2) * s2 +
                    (m1 * lc1 + m2 * l1) * g * s1 + m2 * g * lc2 * s12;

        double V2 = m2 * l1 * lc2 * dq1 * dq1 * s2 + m2 * g * lc2 * s12;

        // Generalized forces acting on the joints
        double f1 = tau1 - V1;
        double f2 = tau2 - V2;

        // Solve for accelerations (M * ddq = f) using 2x2 matrix determinant inversion
        double det = M11 * M22 - M12 * M21;

        if (std::abs(det) < 1e-9)
        {
            det = (det >= 0) ? 1e-9 : -1e-9;
        }

        double ddq1 = (M22 * f1 - M12 * f2) / det;
        double ddq2 = (-M21 * f1 + M11 * f2) / det;

        // 1. Set up inverse Mass Matrix
        Mat2 M_inv;
        M_inv << M22 / det, -M12 / det, -M21 / det, M11 / det;

        // 2. Allocate Jacobian for accelerations w.r.t parameters (2 joints x 6 parameters)
        Eigen::Matrix<double, 2, 6> d_ddq_d_theta;

        // Loop through each parameter to compute partials: [m1, m2, l1, l2, I1, I2]
        for (int i = 0; i < 6; ++i)
        {
            Mat2 dM = Mat2::Zero();
            Vec2 dV = Vec2::Zero();

            switch (i)
            {
            case 0: // m1
                dM(0, 0) = lc1 * lc1;
                dV(0)    = lc1 * g * s1;
                break;
            case 1: // m2
                dM(0, 0) = l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * c2;
                dM(0, 1) = dM(1, 0) = lc2 * lc2 + l1 * lc2 * c2;
                dM(1, 1)            = lc2 * lc2;
                dV(0) = -l1 * lc2 * dq2 * (2.0 * dq1 + dq2) * s2 + l1 * g * s1 +
                        g * lc2 * s12;
                dV(1) = l1 * lc2 * dq1 * dq1 * s2 + g * lc2 * s12;
                break;
            case 2: // l1 (Note: lc1 also depends on l1 -> d(lc1)/dl1 = 0.5)
                dM(0, 0) =
                  2.0 * m1 * lc1 * 0.5 + m2 * (2.0 * l1 + 2.0 * lc2 * c2);
                dM(0, 1) = dM(1, 0) = m2 * lc2 * c2;
                dV(0)               = -m2 * lc2 * dq2 * (2.0 * dq1 + dq2) * s2 +
                        (m1 * 0.5 + m2) * g * s1;
                dV(1) = m2 * lc2 * dq1 * dq1 * s2;
                break;
            case 3: // l2 (Note: lc2 depends on l2 -> d(lc2)/dl2 = 0.5)
                dM(0, 0) = m2 * (2.0 * lc2 * 0.5 + 2.0 * l1 * 0.5 * c2);
                dM(0, 1) = dM(1, 0) = m2 * (2.0 * lc2 * 0.5 + l1 * 0.5 * c2);
                dM(1, 1)            = m2 * (2.0 * lc2 * 0.5);
                dV(0) = -m2 * l1 * 0.5 * dq2 * (2.0 * dq1 + dq2) * s2 +
                        m2 * g * 0.5 * s12;
                dV(1) = m2 * l1 * 0.5 * dq1 * dq1 * s2 + m2 * g * 0.5 * s12;
                break;
            case 4: // I1
                dM(0, 0) = 1.0;
                break;
            case 5: // I2
                dM(0, 0) = 1.0;
                dM(0, 1) = 1.0;
                dM(1, 0) = 1.0;
                dM(1, 1) = 1.0;
                break;
            }

            // Accel derivative: d_ddq = -M^-1 * (dM * ddq + dV)
            Vec2 ddq(ddq1, ddq2);
            d_ddq_d_theta.col(i) = -M_inv * (dM * ddq + dV);
        }

        // 3. Propagate through Semi-Implicit Euler integration
        // state = [q1, q2, dq1, dq2]
        Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();

        // Velocity rows: next_dq = dq + ddq * dt  -> d(next_dq)/dtheta = d(ddq)/dtheta * dt
        J.row(2) = d_ddq_d_theta.row(0) * _dt;
        J.row(3) = d_ddq_d_theta.row(1) * _dt;

        // Position rows: next_q = q + next_dq * dt -> d(next_q)/dtheta = d(next_dq)/dtheta * dt
        J.row(0) = J.row(2) * _dt;
        J.row(1) = J.row(3) * _dt;

        return J;
    }

    void update_parameters(const VecX &theta) override
    {
        _params = theta;
    }

  protected:
    const uint   _nq = 4;
    const uint   _nu = 2;
    const double _dt;

    Vec6 _params;
};

class DoublePendulumTipCost : public stagewise::StageCost
{
  public:
    DoublePendulumTipCost(const Vec2 &target, const double l1, const double l2,
                          const double w_pos, const std::string &name = "")
      : StageCost(name)
      , _target(target)
      , _l1(l1)
      , _l2(l2)
      , _w_pos(w_pos)
    {
    }

    // Evaluate function: Returns a 1x1 vector containing the scalar cost
    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        const double q1 = x(0);
        const double q2 = x(1);

        // Forward kinematics for the tip
        double x_tip = _l1 * std::sin(q1) + _l2 * std::sin(q1 + q2);
        double y_tip = -_l1 * std::cos(q1) - _l2 * std::cos(q1 + q2);

        // Position error
        double dx = x_tip - _target(0);
        double dy = y_tip - _target(1);

        // Pure quadratic tracking cost
        VecX cost(1);
        cost(0) = 0.5 * _w_pos * (dx * dx + dy * dy);
        return cost;
    }

    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        dx.setZero(4); // State dimension is 4 (q1, q2, dq1, dq2)
        du.setZero(u.size());

        const double q1 = x(0);
        const double q2 = x(1);

        double s1  = std::sin(q1);
        double c1  = std::cos(q1);
        double s12 = std::sin(q1 + q2);
        double c12 = std::cos(q1 + q2);

        // Kinematics and tracking error
        double x_tip = _l1 * s1 + _l2 * s12;
        double y_tip = -_l1 * c1 - _l2 * c12;

        double err_x = x_tip - _target(0);
        double err_y = y_tip - _target(1);

        // Kinematic Jacobian components (d(x_tip, y_tip) / d(q1, q2))
        double dx_dq1 = _l1 * c1 + _l2 * c12;
        double dx_dq2 = _l2 * c12;
        double dy_dq1 = _l1 * s1 + _l2 * s12;
        double dy_dq2 = _l2 * s12;

        // Chain rule for state gradient (only affects positions q1, q2)
        dx(0) = _w_pos * (err_x * dx_dq1 + err_y * dy_dq1);
        dx(1) = _w_pos * (err_x * dx_dq2 + err_y * dy_dq2);
        // dx(2) and dx(3) remain 0 (no velocity tracking penalty)
        // du remains 0 (no control penalty)
    }

    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        Hxx.setZero(4, 4);

        const double q1 = x(0);
        const double q2 = x(1);

        double s1  = std::sin(q1);
        double c1  = std::cos(q1);
        double s12 = std::sin(q1 + q2);
        double c12 = std::cos(q1 + q2);

        double err_x = (_l1 * s1 + _l2 * s12) - _target(0);
        double err_y = (-_l1 * c1 - _l2 * c12) - _target(1);

        // Kinematic Jacobians
        double dx_dq1 = _l1 * c1 + _l2 * c12;
        double dx_dq2 = _l2 * c12;
        double dy_dq1 = _l1 * s1 + _l2 * s12;
        double dy_dq2 = _l2 * s12;

        // Second derivatives of position coordinates w.r.t angles
        double d2x_dq1q1 = -_l1 * s1 - _l2 * s12;
        double d2x_dq1q2 = -_l2 * s12;
        double d2x_dq2q2 = -_l2 * s12;

        double d2y_dq1q1 = _l1 * c1 + _l2 * c12;
        double d2y_dq1q2 = _l2 * c12;
        double d2y_dq2q2 = _l2 * c12;

        // Gauss-Newton term (J^T * J) + Exact curvature term
        Hxx(0, 0) = _w_pos * ((dx_dq1 * dx_dq1 + dy_dq1 * dy_dq1) +
                              (err_x * d2x_dq1q1 + err_y * d2y_dq1q1));
        Hxx(0, 1) = _w_pos * ((dx_dq1 * dx_dq2 + dy_dq1 * dy_dq2) +
                              (err_x * d2x_dq1q2 + err_y * d2y_dq1q2));
        Hxx(1, 0) = Hxx(0, 1);
        Hxx(1, 1) = _w_pos * ((dx_dq2 * dx_dq2 + dy_dq2 * dy_dq2) +
                              (err_x * d2x_dq2q2 + err_y * d2y_dq2q2));

        // Rows/cols 2 and 3 remain 0 because velocities are unconstrained
    }

    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        Hxu.setZero(4, u.size());
    }

    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        Huu.setZero(u.size(), u.size());
    }

  private:
    Vec2   _target;
    double _l1;
    double _l2;
    double _w_pos;
};

}
}
}
}