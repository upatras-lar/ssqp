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
    DoublePendulumAmbientDynamics(const Eigen::Vector<double, 9> &params,
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
                          _g_m1lc1_m2l1 * s1 + _g_m2lc2 * s12 + _b1 * dq1;

        const double V2 =
          _m2_l1_lc2 * dq1 * dq1 * s2 + _g_m2lc2 * s12 + _b2 * dq2;

        Eigen::LDLT<Mat2> ldlt(M);
        const Vec2        rhs = Vec2(tau1 - V1, tau2 - V2);
        const Vec2        ddq = ldlt.solve(rhs);

        const double ndq1 = dq1 + ddq[0] * _dt;
        const double ndq2 = dq2 + ddq[1] * _dt;

        const double d1 = ndq1 * _dt, d2 = ndq2 * _dt;
        const double cd1 = std::cos(d1), sd1 = std::sin(d1);
        const double cd2 = std::cos(d2), sd2 = std::sin(d2);

        VecX nx(_nx);
        nx << s1 * cd1 + c1 * sd1, c1 * cd1 - s1 * sd1, s2 * cd2 + c2 * sd2,
          c2 * cd2 - s2 * sd2, ndq1, ndq2;

        double norm1  = std::sqrt(nx(0) * nx(0) + nx(1) * nx(1));
        double norm2  = std::sqrt(nx(2) * nx(2) + nx(3) * nx(3));
        nx(0)        /= norm1;
        nx(1)        /= norm1;
        nx(2)        /= norm2;
        nx(3)        /= norm2;

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
                          _g_m1lc1_m2l1 * s1 + _g_m2lc2 * s12 + _b1 * dq1;
        const double V2 =
          _m2_l1_lc2 * dq1 * dq1 * s2 + _g_m2lc2 * s12 + _b2 * dq2;

        Eigen::LDLT<Mat2> ldlt(M);
        const Vec2        rhs = Vec2(tau1 - V1, tau2 - V2);
        const Vec2        ddq = ldlt.solve(rhs);

        // 1. w.r.t. q1 (damping has no position dependence)
        const Vec2 dV_dq1(_g_m1lc1_m2l1 * c1 + _g_m2lc2 * c12, _g_m2lc2 * c12);

        // 2. w.r.t. q2 (damping has no position dependence)
        Mat2 dM_dq2;
        dM_dq2(0, 0) = -2.0 * _m2_l1_lc2 * s2;
        dM_dq2(0, 1) = dM_dq2(1, 0) = -_m2_l1_lc2 * s2;
        dM_dq2(1, 1)                = 0.0;

        const Vec2 dV_dq2(-_m2_l1_lc2 * dq2 * (2.0 * dq1 + dq2) * c2 +
                            _g_m2lc2 * c12,
                          _m2_l1_lc2 * dq1 * dq1 * c2 + _g_m2lc2 * c12);

        // 3. w.r.t. dq1 -- damping adds +_b1 to the joint-1 entry
        const Vec2 dV_ddq1(-2.0 * _m2_l1_lc2 * dq2 * s2 + _b1,
                           2.0 * _m2_l1_lc2 * dq1 * s2);

        // 4. w.r.t. dq2 -- damping adds +_b2 to the joint-2 entry
        const Vec2 dV_ddq2(-2.0 * _m2_l1_lc2 * (dq1 + dq2) * s2, _b2);

        // --- Tangent-space acceleration Jacobians ---
        Mat2 ddq_dq, ddq_ddq;
        ddq_dq.col(0)  = ldlt.solve(-dV_dq1);
        ddq_dq.col(1)  = ldlt.solve(-dV_dq2 - dM_dq2 * ddq);
        ddq_ddq.col(0) = ldlt.solve(-dV_ddq1);
        ddq_ddq.col(1) = ldlt.solve(-dV_ddq2);

        const Mat2 I         = Mat2::Identity();
        dX.block<2, 2>(0, 0) = I + ddq_dq * (_dt * _dt);
        dX.block<2, 2>(0, 2) = I * _dt + ddq_ddq * (_dt * _dt);
        dX.block<2, 2>(2, 0) = ddq_dq * _dt;
        dX.block<2, 2>(2, 2) = I + ddq_ddq * _dt;

        // --- Control Jacobian: depends on which joint(s) are actuated ---
        switch (_mode)
        {
        case ActuationMode::Full:
        {
            Mat2 I = Mat2::Identity();

            Mat2 Minv = ldlt.solve(I);

            dU.block<2, 2>(0, 0) = Minv * (_dt * _dt);
            dU.block<2, 2>(2, 0) = Minv * _dt;
            break;
        }
        case ActuationMode::Pendubot:
        {
            const Vec2 e1        = Vec2(1.0, 0.0);
            const Vec2 ddq_dtau1 = ldlt.solve(e1);

            dU.block<2, 1>(0, 0) = ddq_dtau1 * (_dt * _dt);
            dU.block<2, 1>(2, 0) = ddq_dtau1 * _dt;
            break;
        }
        case ActuationMode::Acrobot:
        {
            const Vec2 e2        = Vec2(0.0, 1.0);
            const Vec2 ddq_dtau2 = ldlt.solve(e2);

            dU.block<2, 1>(0, 0) = ddq_dtau2 * (_dt * _dt);
            dU.block<2, 1>(2, 0) = ddq_dtau2 * _dt;
            break;
        }
        }
    }

    void hessprod_xx(const VecX &, const VecX &, const VecX &,
                     MatX &Pxx) const override
    {
        // Damping is exactly linear in dq, so it adds no curvature here.
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

    /**
     * @brief Computes the Regressor Matrix (Y) and Measured Torques (tau) 
     * * Parameter Vector order: 
     * Phi = [ m1*r1, I1, cf1, b1, m2*r2, m2, I2, cf2, b2 ]  (Size: 9)
     * * @param x       Ambient State vector [sin(q1), cos(q1), sin(q2), cos(q2), dq1, dq2]
     * @param ddq     Calculated/measured joint accelerations [ddq1, ddq2]
     * @param u       Control/measured torque input
     * @param Y       Output Regressor Matrix (2 rows x 9 columns)
     * @param tau     Output Torque vector (Size: 2)
     */
    void compute_regressor_and_torque(const VecX &x, const Vec2 &ddq,
                                      const VecX                  &u,
                                      Eigen::Matrix<double, 2, 9> &Y,
                                      Vec2                        &tau) const
    {
        // 1. Extract states from ambient vector
        const double s1 = x(0), c1 = x(1);
        const double s2 = x(2), c2 = x(3);
        const double dq1 = x(4), dq2 = x(5);
        const double ddq1 = ddq(0), ddq2 = ddq(1);

        // 2. Resolve target torques based on actuation pattern
        double tau1_meas = 0.0, tau2_meas = 0.0;
        get_torques(u, tau1_meas, tau2_meas);
        tau << tau1_meas, tau2_meas;

        // Constants
        const double g = 9.81;
        const double l1 =
          _params[2]; // Fixed geometric length from known parameters

        // Precompute trig combinations matching manipulator equations
        const double s12 = s1 * c2 + c1 * s2;

        // 3. Initialize Regressor Matrix to Zero
        Y.setZero();

        // --- ROW 1: Joint 1 Dynamics ---
        Y(0, 0) = -g * s1;                // m1*r1 (Gravity)
        Y(0, 1) = ddq1;                   // I1 (Inertia)
        Y(0, 2) = std::atan(100.0 * dq1); // cf1 (Coulomb Friction)
        Y(0, 3) = dq1;                    // b1 (Viscous Friction)
        Y(0, 4) = 2.0 * l1 * c2 * ddq1 + l1 * c2 * ddq2 -
                  2.0 * dq2 * l1 * s2 * dq1 - dq2 * dq2 * l1 * s2 -
                  g * s12;                      // m2*r2
        Y(0, 5) = l1 * l1 * ddq1 - g * l1 * s1; // m2
        Y(0, 6) = ddq1 + ddq2;                  // I2
        Y(0, 7) = 0.0;                          // cf2 (No impact on Joint 1)
        Y(0, 8) = 0.0;                          // b2 (No impact on Joint 1)

        // --- ROW 2: Joint 2 Dynamics ---
        Y(1, 0) = 0.0; // m1*r1 (No impact on Joint 2)
        Y(1, 1) = 0.0; // I1 (No impact on Joint 2)
        Y(1, 2) = 0.0; // cf1 (No impact on Joint 2)
        Y(1, 3) = 0.0; // b1
        Y(1, 4) = l1 * c2 * ddq1 + l1 * s2 * dq1 * dq1 - g * s12; // m2*r2
        Y(1, 5) = 0.0;                                            // m2
        Y(1, 6) = ddq1 + ddq2;                                    // I2
        Y(1, 7) = std::atan(100.0 * dq2);                         // cf2
        Y(1, 8) = dq2;                                            // b2
    }

    uint parameter_size() const override
    {
        return 9;
    }

    VecX parameters() const override
    {
        return _params;
    }

    void update_parameters(const VecX &theta) override
    {
        _params = theta;

        // Extract exact symbols from the identified vector
        const double m1r1 = _params[0]; // Phi[0]
        const double I1   = _params[1]; // Phi[1]
        const double cf1  = _params[2]; // Phi[2] (Coulomb friction, if used)
        const double b1   = _params[3]; // Phi[3]
        const double m2r2 = _params[4]; // Phi[4]
        const double m2   = _params[5]; // Phi[5]
        const double I2   = _params[6]; // Phi[6]
        const double cf2  = _params[7]; // Phi[7]
        const double b2   = _params[8]; // Phi[8]

        const double g  = 9.81;
        const double l1 = 0.05; // Known fixed geometric link length
        const double l2 = 0.05; // Known fixed geometric link length

        // Implicit center of mass locations
        const double rc1 = 0.5 * l1;
        const double rc2 = 0.5 * l2;

        // Update internal damping fields
        _b1 = b1;
        _b2 = b2;

        // Direct mathematical substitution
        _m2_l1_lc2   = l1 * m2r2; // m2 * l1 * lc2
        _m2_l1_lc2_2 = 2.0 * _m2_l1_lc2;

        // Explicitly mapped inertia matrix combinations
        _M11_const = I1 + I2 + (m1r1 * rc1) + m2 * (l1 * l1) + (m2r2 * rc2);
        _M12_const = I2 + (m2r2 * rc2); // I2 + m2 * lc2^2
        _M22_const = I2 + (m2r2 * rc2); // I2 + m2 * lc2^2

        // Gravity vector components
        _g_m1lc1_m2l1 = (m1r1 + m2 * l1) * g; // (m1*lc1 + m2*l1) * g
        _g_m2lc2      = m2r2 * g;             // m2 * lc2 * g
    }

    // Which actuation pattern this instance represents.
    ActuationMode actuation_mode() const
    {
        return _mode;
    }

    // Viscous joint damping coefficients [b1, b2]. Fixed property of the
    // model, independent of the 6-parameter vector above.
    void set_damping(const Vec2 &damping)
    {
        _b1 = damping(0);
        _b2 = damping(1);
    }
    Vec2 damping() const
    {
        return Vec2(_b1, _b2);
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

    const uint               _nx  = 6;
    const uint               _ndx = 4;
    const ActuationMode      _mode;
    const uint               _nu;
    const double             _dt;
    double                   _b1; // joint 1 viscous damping coefficient
    double                   _b2; // joint 2 viscous damping coefficient
    Eigen::Vector<double, 9> _params;

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

  protected:
    const uint _ndx = 4; // Tangent size

    Vec6 _x_des;
    Vec4 _q_weight;
};

}
}
}
}