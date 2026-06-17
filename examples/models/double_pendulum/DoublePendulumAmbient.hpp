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
/* Enum used to select the mode of actuation for the double pendulum ambient
 * model
 *
 * Options:
 *   - Full : both joints actuated (classic double pendulum)
 *   - Pendubot       : only the first joint actuated, second passive
 *   - Acrobot        : only the second joint actuated, first passive
 */
enum ActuationMode
{
    Full,
    Pendubot,
    Acrobot
};

/* Two-Link Pendulum Ambient Dynamics
 * Represents the system's state in ambient space: (sin(a), cos(a))
 * for each angle a.
 *
 * A single class spanning all three actuation patterns of a two-link
 * pendulum, selected via ActuationMode at construction time.
 *
 * Defines the dynamics relative to 6 parameters:
 *  1. Mass 1 (m1)
 *  2. Mass 2 (m2)
 *  3. Length 1 (l1)
 *  4. Length 2 (l2)
 *  5. Inertia 1 (I1)
 *  6. Inertia 2 (I2)
 */
class DoublePendulumAmbientDynamics : public stagewise::StageDynamics
{
  public:
    DoublePendulumAmbientDynamics(const Vec6   &params,
                                  ActuationMode mode      = ActuationMode::Full,
                                  const double dt         = 0.01,
                                  const std::string &name = "")
      : StageDynamics(name)
      , _mode(mode)
      , _nu(mode == ActuationMode::Full ? 2 : 1)
      , _dt(dt)
    {
        update_parameters(params);
    }

    uint output_size() const override
    {
        return _nx;
    }
    uint state_size() const override
    {
        return _nx;
    }
    uint state_diff_size() const override
    {
        return _ndx;
    }
    uint control_size() const override
    {
        return _nu;
    }

    // x  = [s1, c1, s2, c2, dq1, dq2]  (size _nq  = 6, ambient)
    // dx = [δq1, δq2, δdq1, δdq2]       (size _ndq = 4, tangent)
    // y  = x ⊞ dx
    VecX state_right_plus(const VecX &x, const VecX &dx) const override
    {
        const double s1 = x(0), c1 = x(1);
        const double s2 = x(2), c2 = x(3);
        const double dq1 = x(4), dq2 = x(5);

        const double dth1 = dx(0), dth2 = dx(1);
        const double ddq1 = dx(2), ddq2 = dx(3);

        const double cd1 = std::cos(dth1), sd1 = std::sin(dth1);
        const double cd2 = std::cos(dth2), sd2 = std::sin(dth2);

        VecX y(_nx);
        y << s1 * cd1 + c1 * sd1, c1 * cd1 - s1 * sd1, s2 * cd2 + c2 * sd2,
          c2 * cd2 - s2 * sd2, dq1 + ddq1, dq2 + ddq2;
        return y;
    }

    // x, y = [s1, c1, s2, c2, dq1, dq2]  (size _nq  = 6, ambient)
    VecX state_right_minus(const VecX &x, const VecX &y) const override
    {
        const double xs1 = x(0), xc1 = x(1);
        const double xs2 = x(2), xc2 = x(3);
        const double ys1 = y(0), yc1 = y(1);
        const double ys2 = y(2), yc2 = y(3);

        VecX dx(_ndx);
        dx << std::atan2(xs1 * yc1 - xc1 * ys1, xc1 * yc1 + xs1 * ys1),
          std::atan2(xs2 * yc2 - xc2 * ys2, xc2 * yc2 + xs2 * ys2), x(4) - y(4),
          x(5) - y(5);
        return dx;
    }

    // State layout: [sin(q1), cos(q1), sin(q2), cos(q2), dq1, dq2]
    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        const double s1 = x(0), c1 = x(1);
        const double s2 = x(2), c2 = x(3);
        const double dq1 = x(4), dq2 = x(5);

        double tau1, tau2;
        get_torques(u, tau1, tau2);

        const double s12 = s1 * c2 + c1 * s2;

        Mat2 M;
        M(0, 0) = _M11_const + _m2_l1_lc2_2 * c2;
        M(0, 1) = _M12_const + _m2_l1_lc2 * c2;
        M(1, 0) = M(0, 1);
        M(1, 1) = _M22_const;

        const double V1 = -_m2_l1_lc2 * dq2 * (2.0 * dq1 + dq2) * s2 +
                          _g_m1lc1_m2l1 * s1 + _g_m2lc2 * s12;
        const double V2 = _m2_l1_lc2 * dq1 * dq1 * s2 + _g_m2lc2 * s12;

        const Mat2 M_inv = M.inverse();
        const Vec2 ddq   = M_inv * Vec2(tau1 - V1, tau2 - V2);

        const double ndq1 = dq1 + ddq[0] * _dt;
        const double ndq2 = dq2 + ddq[1] * _dt;

        const double d1 = ndq1 * _dt, d2 = ndq2 * _dt;
        const double cd1 = std::cos(d1), sd1 = std::sin(d1);
        const double cd2 = std::cos(d2), sd2 = std::sin(d2);

        VecX nx(_nx);
        nx << s1 * cd1 + c1 * sd1, c1 * cd1 - s1 * sd1, s2 * cd2 + c2 * sd2,
          c2 * cd2 - s2 * sd2, ndq1, ndq2;
        return nx;
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        dX.setZero(_ndx, _ndx);
        dU.setZero(_ndx, _nu);

        const double s1 = x(0), c1 = x(1);
        const double s2 = x(2), c2 = x(3);
        const double dq1 = x(4), dq2 = x(5);

        double tau1, tau2;
        get_torques(u, tau1, tau2);

        const double s12 = s1 * c2 + c1 * s2;
        const double c12 = c1 * c2 - s1 * s2;

        // --- Recompute forward dynamics ---
        Mat2 M;
        M(0, 0) = _M11_const + _m2_l1_lc2_2 * c2;
        M(0, 1) = _M12_const + _m2_l1_lc2 * c2;
        M(1, 0) = M(0, 1);
        M(1, 1) = _M22_const;

        const double V1 = -_m2_l1_lc2 * dq2 * (2.0 * dq1 + dq2) * s2 +
                          _g_m1lc1_m2l1 * s1 + _g_m2lc2 * s12;
        const double V2 = _m2_l1_lc2 * dq1 * dq1 * s2 + _g_m2lc2 * s12;

        const Mat2 M_inv = M.inverse();
        const Vec2 ddq   = M_inv * Vec2(tau1 - V1, tau2 - V2);

        // 1. w.r.t. q1
        const Vec2 dV_dq1(_g_m1lc1_m2l1 * c1 + _g_m2lc2 * c12, _g_m2lc2 * c12);

        // 2. w.r.t. q2
        Mat2 dM_dq2;
        dM_dq2(0, 0) = -2.0 * _m2_l1_lc2 * s2;
        dM_dq2(0, 1) = dM_dq2(1, 0) = -_m2_l1_lc2 * s2;
        dM_dq2(1, 1)                = 0.0;

        const Vec2 dV_dq2(-_m2_l1_lc2 * dq2 * (2.0 * dq1 + dq2) * c2 +
                            _g_m2lc2 * c12,
                          _m2_l1_lc2 * dq1 * dq1 * c2 + _g_m2lc2 * c12);

        // 3. w.r.t. dq1
        const Vec2 dV_ddq1(-2.0 * _m2_l1_lc2 * dq2 * s2,
                           2.0 * _m2_l1_lc2 * dq1 * s2);

        // 4. w.r.t. dq2
        const Vec2 dV_ddq2(-2.0 * _m2_l1_lc2 * (dq1 + dq2) * s2, 0.0);

        // --- Tangent-space acceleration Jacobians ---
        Mat2 ddq_dq, ddq_ddq;
        ddq_dq.col(0)  = M_inv * (-dV_dq1);
        ddq_dq.col(1)  = M_inv * (-dV_dq2 - dM_dq2 * ddq);
        ddq_ddq.col(0) = M_inv * (-dV_ddq1);
        ddq_ddq.col(1) = M_inv * (-dV_ddq2);

        const Mat2 I         = Mat2::Identity();
        dX.block<2, 2>(0, 0) = I + ddq_dq * (_dt * _dt);
        dX.block<2, 2>(0, 2) = I * _dt + ddq_ddq * (_dt * _dt);
        dX.block<2, 2>(2, 0) = ddq_dq * _dt;
        dX.block<2, 2>(2, 2) = I + ddq_ddq * _dt;

        // --- Control Jacobian: depends on which joint(s) are actuated ---
        switch (_mode)
        {
        case ActuationMode::Full:
            dU.block<2, 2>(0, 0) = M_inv * (_dt * _dt);
            dU.block<2, 2>(2, 0) = M_inv * _dt;
            break;
        case ActuationMode::Pendubot:
        {
            const Vec2 ddq_dtau1 = M_inv.col(0);
            dU.block<2, 1>(0, 0) = ddq_dtau1 * (_dt * _dt);
            dU.block<2, 1>(2, 0) = ddq_dtau1 * _dt;
            break;
        }
        case ActuationMode::Acrobot:
        {
            const Vec2 ddq_dtau2 = M_inv.col(1);
            dU.block<2, 1>(0, 0) = ddq_dtau2 * (_dt * _dt);
            dU.block<2, 1>(2, 0) = ddq_dtau2 * _dt;
            break;
        }
        }
    }

    void hessprod_xx(const VecX &, const VecX &, const VecX &,
                     MatX &Pxx) const override
    {
        Pxx.setZero(_ndx, _ndx);
    }
    void hessprod_xu(const VecX &, const VecX &, const VecX &,
                     MatX &Pxu) const override
    {
        Pxu.setZero(_ndx, _nu);
    }
    void hessprod_uu(const VecX &, const VecX &, const VecX &,
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

    MatX parameter_gradient(const VecX &x, const VecX &u) const override
    {
        const double s1 = x(0), c1 = x(1);
        const double s2 = x(2), c2 = x(3);
        const double dq1 = x(4), dq2 = x(5);

        double tau1, tau2;
        get_torques(u, tau1, tau2);

        const double s12 = s1 * c2 + c1 * s2;

        // Reconstruct using cache variables where applicable
        const double M11 = _M11_const + _m2_l1_lc2_2 * c2;
        const double M12 = _M12_const + _m2_l1_lc2 * c2;
        const double M22 = _M22_const;

        const double V1 = -_m2_l1_lc2 * dq2 * (2.0 * dq1 + dq2) * s2 +
                          _g_m1lc1_m2l1 * s1 + _g_m2lc2 * s12;
        const double V2 = _m2_l1_lc2 * dq1 * dq1 * s2 + _g_m2lc2 * s12;

        double det = M11 * M22 - M12 * M12;
        if (std::abs(det) < 1e-9) det = (det >= 0) ? 1e-9 : -1e-9;

        Mat2 M_inv;
        M_inv << M22 / det, -M12 / det, -M12 / det, M11 / det;

        const Vec2 ddq = M_inv * Vec2(tau1 - V1, tau2 - V2);

        Eigen::Matrix<double, 2, 6> d_ddq_d_theta;

        // Recompute localized elements for partial parameter derivatives
        const double m1 = _params[0], m2 = _params[1];
        const double l1 = _params[2], l2 = _params[3];
        const double g   = 9.81;
        const double lc1 = 0.5 * l1, lc2 = 0.5 * l2;

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
            case 2: // l1
                dM(0, 0) =
                  2.0 * m1 * lc1 * 0.5 + m2 * (2.0 * l1 + 2.0 * lc2 * c2);
                dM(0, 1) = dM(1, 0) = m2 * lc2 * c2;
                dV(0)               = -m2 * lc2 * dq2 * (2.0 * dq1 + dq2) * s2 +
                        (m1 * 0.5 + m2) * g * s1;
                dV(1) = m2 * lc2 * dq1 * dq1 * s2;
                break;
            case 3: // l2
                dM(0, 0) = m2 * (lc2 + l1 * c2);
                dM(0, 1) = dM(1, 0) = m2 * (lc2 + l1 * 0.5 * c2);
                dM(1, 1)            = m2 * lc2;
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

            d_ddq_d_theta.col(i) = -M_inv * (dM * ddq + dV);
        }

        Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
        J.row(2)                      = d_ddq_d_theta.row(0) * _dt;
        J.row(3)                      = d_ddq_d_theta.row(1) * _dt;
        J.row(0)                      = J.row(2) * _dt;
        J.row(1)                      = J.row(3) * _dt;
        return J;
    }

    void update_parameters(const VecX &theta) override
    {
        _params = theta;

        const double m1 = _params[0], m2 = _params[1];
        const double l1 = _params[2], l2 = _params[3];
        const double I1 = _params[4], I2 = _params[5];
        const double g   = 9.81;
        const double lc1 = 0.5 * l1, lc2 = 0.5 * l2;

        // Cache parameters bundle constants
        _m2_l1_lc2   = m2 * l1 * lc2;
        _m2_l1_lc2_2 = 2.0 * _m2_l1_lc2;

        _M11_const = I1 + I2 + m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2);
        _M12_const = I2 + m2 * lc2 * lc2;
        _M22_const = I2 + m2 * lc2 * lc2;

        _g_m1lc1_m2l1 = (m1 * lc1 + m2 * l1) * g;
        _g_m2lc2      = m2 * g * lc2;
    }

    // Which actuation pattern this instance represents.
    ActuationMode actuation_mode() const
    {
        return _mode;
    }

  protected:
    // Resolves (tau1, tau2) from the control vector according to _mode.
    // `u` only ever holds the *actuated* torque(s): size 2 for
    // FULLY_ACTUATED, size 1 for PENDUBOT / ACROBOT.
    void get_torques(const VecX &u, double &tau1, double &tau2) const
    {
        switch (_mode)
        {
        case ActuationMode::Full:
            tau1 = u(0);
            tau2 = u(1);
            break;
        case ActuationMode::Pendubot:
            tau1 = u(0);
            tau2 = 0.0;
            break;
        case ActuationMode::Acrobot:
            tau1 = 0.0;
            tau2 = u(0);
            break;
        }
    }

    const uint          _nx  = 6;
    const uint          _ndx = 4;
    const ActuationMode _mode;
    const uint          _nu;
    const double        _dt;
    Vec6                _params;

    // Invariant cached calculation bundles
    double _m2_l1_lc2;
    double _m2_l1_lc2_2;
    double _M11_const;
    double _M12_const;
    double _M22_const;
    double _g_m1lc1_m2l1;
    double _g_m2lc2;
};

class DoublePendulumAmbientTrackingCost : public stagewise::StageCost
{
  public:
    DoublePendulumAmbientTrackingCost(const Vec6 &x_des, const Vec4 &q_weight)
      : _x_des(x_des)
      , _q_weight(q_weight)
    {
    }

    void set_target(const Vec6 &x_des)
    {
        _x_des = x_des;
    }

    // Evaluates: 0.5 * dx^T * Q * dx
    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        // 1. Compute tangent space error dx = x ⊖ x_des
        const double xs1 = x(0), xc1 = x(1);
        const double xs2 = x(2), xc2 = x(3);

        const double ys1 = _x_des(0), yc1 = _x_des(1);
        const double ys2 = _x_des(2), yc2 = _x_des(3);

        const double dq1 =
          std::atan2(xs1 * yc1 - xc1 * ys1, xc1 * yc1 + xs1 * ys1);
        const double dq2 =
          std::atan2(xs2 * yc2 - xc2 * ys2, xc2 * yc2 + xs2 * ys2);
        const double ddq1 = x(4) - _x_des(4);
        const double ddq2 = x(5) - _x_des(5);

        // 2. Compute quadratic cost
        double cost =
          0.5 * (_q_weight(0) * dq1 * dq1 + _q_weight(1) * dq2 * dq2 +
                 _q_weight(2) * ddq1 * ddq1 + _q_weight(3) * ddq2 * ddq2);

        VecX result(1);
        result(0) = cost;
        return result;
    }

    // Gradient computed directly in the tangent space
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        dx.resize(_ndx); // size 4
        du.setZero(u.size());

        // Tangent space position/velocity errors
        const double dq1 = std::atan2(x(0) * _x_des(1) - x(1) * _x_des(0),
                                      x(1) * _x_des(1) + x(0) * _x_des(0));
        const double dq2 = std::atan2(x(2) * _x_des(3) - x(3) * _x_des(2),
                                      x(3) * _x_des(3) + x(2) * _x_des(2));

        // dx = Q * error
        dx(0) = _q_weight(0) * dq1;
        dx(1) = _q_weight(1) * dq2;
        dx(2) = _q_weight(2) * (x(4) - _x_des(4));
        dx(3) = _q_weight(3) * (x(5) - _x_des(5));
    }

    // Constant diagonal Hessian block for state error
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        Hxx.setZero(_ndx, _ndx);
        Hxx.diagonal() = _q_weight;
    }

    // State-control cross terms are zero
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        Hxu.setZero(_ndx, u.size());
    }

    // Constant diagonal Hessian block for control effort
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        Huu.setZero(u.size(), u.size());
    }

  private:
    const uint _ndx = 4; // Tangent size

    Vec6 _x_des;
    Vec4 _q_weight;
};

}
}
}
}