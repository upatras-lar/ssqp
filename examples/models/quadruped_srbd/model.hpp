#include "types.hpp"
#include "model/Model.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageConstraints.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

using namespace ssqp;
using namespace ssqp::model::stagewise;

// ---------------------------------------------------------------------------
// Math Utilities
// ---------------------------------------------------------------------------
namespace utils
{
inline Eigen::Matrix3d skew(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    S << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
    return S;
}

// Assumes Pinocchio/Eigen layout: x, y, z, w where w is at index 3
inline Eigen::Matrix3d quat_to_rotation_matrix(const Eigen::Vector4d &q)
{
    Eigen::Quaterniond quat(q(3), q(0), q(1), q(2));
    return quat.toRotationMatrix();
}

inline Eigen::Vector4d quat_integrate(const Eigen::Vector4d &q,
                                      const Eigen::Vector3d &omega)
{
    Eigen::Quaterniond quat(q(3), q(0), q(1), q(2));
    double             angle = omega.norm();
    if (angle < 1e-8) return q;
    Eigen::AngleAxisd  aa(angle, omega / angle);
    Eigen::Quaterniond q_next = quat * Eigen::Quaterniond(aa);
    q_next.normalize();
    return Eigen::Vector4d(q_next.x(), q_next.y(), q_next.z(), q_next.w());
}

inline Eigen::Vector4d quat_mul(const Eigen::Vector4d &q1,
                                const Eigen::Vector4d &q2)
{
    Eigen::Quaterniond quat1(q1(3), q1(0), q1(1), q1(2));
    Eigen::Quaterniond quat2(q2(3), q2(0), q2(1), q2(2));
    Eigen::Quaterniond res = quat1 * quat2;
    return Eigen::Vector4d(res.x(), res.y(), res.z(), res.w());
}
}

// ---------------------------------------------------------------------------
// Offsets & Constants
// ---------------------------------------------------------------------------
constexpr int NF        = 4;
constexpr int _O_P_COM  = 0;
constexpr int _O_THETA  = 3;
constexpr int _O_V_COM  = 6;
constexpr int _O_OMEGA  = 9;
constexpr int _O_P_FEET = 12;

constexpr int _O_F = 0;
constexpr int _O_V = 12;

constexpr double MU      = 0.6;
constexpr double F_Z_MIN = 1.0;
constexpr double F_Z_MAX = 250.0;
constexpr double Z_MIN   = 0.20;

// Parameters
struct SRBDParameters
{
    const double dt;
    const double mass;
    const MatX   inertia;
    MatX         inertia_inv;
    const uint   nq;
    const uint   ndq;
    const uint   nu;

    const Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81);

    double w_p  = 100.0;
    double w_o  = 1.0;
    double w_w  = 1.0;
    double w_pf = 500.0;
    double w_f  = 1e-4;
    double w_vf = 1e-4;
};

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------
class SRBDynamics : public StageDynamics
{
  protected:
    SRBDParameters _params;

  public:
    SRBDynamics(const SRBDParameters &params) : StageDynamics(), _params(params)
    {
        _params.inertia_inv = _params.inertia.inverse();
    }

    uint output_size() const override
    {
        return _params.nq;
    }
    uint state_size() const override
    {
        return _params.nq;
    }
    uint state_diff_size() const override
    {
        return _params.ndq;
    }
    uint control_size() const override
    {
        return _params.nu;
    }

    VecX state_right_plus(const VecX &x, const VecX &dx) const override
    {
        VecX x_new          = VecX::Zero(_params.nq);
        x_new.segment<3>(0) = x.segment<3>(0) + dx.segment<3>(_O_P_COM);
        x_new.segment<4>(3) =
          utils::quat_integrate(x.segment<4>(3), dx.segment<3>(_O_THETA));
        x_new.segment<3>(7)  = x.segment<3>(7) + dx.segment<3>(_O_V_COM);
        x_new.segment<3>(10) = x.segment<3>(10) + dx.segment<3>(_O_OMEGA);
        for (int i = 0; i < NF; ++i)
        {
            x_new.segment<3>(13 + 3 * i) =
              x.segment<3>(13 + 3 * i) + dx.segment<3>(_O_P_FEET + 3 * i);
        }
        return x_new;
    }

    VecX state_right_minus(const VecX &x, const VecX &y) const override
    {
        VecX dx                 = VecX::Zero(_params.ndq);
        dx.segment<3>(_O_P_COM) = x.segment<3>(0) - y.segment<3>(0);

        Eigen::Matrix3d R_y   = utils::quat_to_rotation_matrix(y.segment<4>(3));
        Eigen::Matrix3d R_x   = utils::quat_to_rotation_matrix(x.segment<4>(3));
        Eigen::Matrix3d R_err = R_y.transpose() * R_x;

        double tr    = R_err.trace();
        double angle = std::acos(std::clamp((tr - 1.0) / 2.0, -1.0, 1.0));

        Eigen::Vector3d dtheta = Eigen::Vector3d::Zero();
        if (std::abs(angle) >= 1e-8)
        {
            dtheta << R_err(2, 1) - R_err(1, 2), R_err(0, 2) - R_err(2, 0),
              R_err(1, 0) - R_err(0, 1);
            dtheta *= angle / (2.0 * std::sin(angle));
        }
        dx.segment<3>(_O_THETA) = dtheta;

        dx.segment<3>(_O_V_COM) = x.segment<3>(7) - y.segment<3>(7);
        dx.segment<3>(_O_OMEGA) = x.segment<3>(10) - y.segment<3>(10);

        for (int i = 0; i < NF; ++i)
        {
            dx.segment<3>(_O_P_FEET + 3 * i) =
              x.segment<3>(13 + 3 * i) - y.segment<3>(13 + 3 * i);
        }
        return dx;
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        Eigen::Vector3d p = x.segment<3>(0);
        Eigen::Vector4d q = x.segment<4>(3);
        Eigen::Vector3d v = x.segment<3>(7);
        Eigen::Vector3d w = x.segment<3>(10);

        Eigen::Matrix3d R       = utils::quat_to_rotation_matrix(q);
        Eigen::Matrix3d I_w     = R * _params.inertia * R.transpose();
        Eigen::Matrix3d I_w_inv = R * _params.inertia_inv * R.transpose();

        Eigen::Vector3d f_total   = Eigen::Vector3d::Zero();
        Eigen::Vector3d tau_total = Eigen::Vector3d::Zero();

        for (int i = 0; i < NF; ++i)
        {
            Eigen::Vector3d f_i  = u.segment<3>(_O_F + 3 * i);
            Eigen::Vector3d r_i  = x.segment<3>(13 + 3 * i) - p;
            f_total             += f_i;
            tau_total           += r_i.cross(f_i);
        }

        Eigen::Vector3d v_new =
          v + _params.dt * (f_total / _params.mass + _params.gravity);
        Eigen::Vector3d w_new =
          w + _params.dt * (I_w_inv * (tau_total - w.cross(I_w * w)));
        Eigen::Vector3d p_new = p + _params.dt * v_new;
        Eigen::Vector4d q_new = utils::quat_integrate(q, _params.dt * w_new);

        VecX x_new           = VecX::Zero(_params.nq);
        x_new.segment<3>(0)  = p_new;
        x_new.segment<4>(3)  = q_new;
        x_new.segment<3>(7)  = v_new;
        x_new.segment<3>(10) = w_new;

        for (int i = 0; i < NF; ++i)
        {
            x_new.segment<3>(13 + 3 * i) =
              x.segment<3>(13 + 3 * i) +
              _params.dt * u.segment<3>(_O_V + 3 * i);
        }

        return x_new;
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        Eigen::Vector3d p = x.segment<3>(0);
        Eigen::Vector4d q = x.segment<4>(3);
        Eigen::Vector3d v = x.segment<3>(7);
        Eigen::Vector3d w = x.segment<3>(10);

        Eigen::Matrix3d R       = utils::quat_to_rotation_matrix(q);
        Eigen::Matrix3d I_w     = R * _params.inertia * R.transpose();
        Eigen::Matrix3d I_w_inv = R * _params.inertia_inv * R.transpose();

        Eigen::Vector3d              tau = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> r(NF);
        for (int i = 0; i < NF; ++i)
        {
            r[i]  = x.segment<3>(13 + 3 * i) - p;
            tau  += r[i].cross(u.segment<3>(_O_F + 3 * i));
        }

        Eigen::Vector3d h     = I_w * w;
        Eigen::Vector3d w_new = w + _params.dt * (I_w_inv * (tau - w.cross(h)));
        Eigen::Vector3d phi   = _params.dt * w_new;
        double          angle = phi.norm();

        Eigen::Matrix3d R_dq = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d J_r  = Eigen::Matrix3d::Identity();

        if (angle < 1e-10)
        {
            J_r -= 0.5 * utils::skew(phi);
        }
        else
        {
            Eigen::Vector3d axis = phi / angle;
            Eigen::Matrix3d K    = utils::skew(axis);
            R_dq += std::sin(angle) * K + (1.0 - std::cos(angle)) * (K * K);
            Eigen::Matrix3d S_phi = utils::skew(phi);
            J_r += -((1.0 - std::cos(angle)) / (angle * angle)) * S_phi +
                   ((angle - std::sin(angle)) / (std::pow(angle, 3))) *
                     (S_phi * S_phi);
        }

        Eigen::Matrix3d S_w = utils::skew(w);
        Eigen::Matrix3d S_h = utils::skew(h);

        Eigen::Matrix3d A_ww = Eigen::Matrix3d::Identity() +
                               _params.dt * (I_w_inv * (S_h - S_w * I_w));
        Eigen::Matrix3d dgyro_dtheta =
          (h * w.transpose()) - (w.dot(h) * Eigen::Matrix3d::Identity()) -
          S_w * I_w * S_w;
        Eigen::Matrix3d A_wtheta = _params.dt * (I_w_inv * dgyro_dtheta);

        Eigen::Vector3d              dtau_dp_com_vec = Eigen::Vector3d::Zero();
        Eigen::Matrix3d              dtau_dp_com     = Eigen::Matrix3d::Zero();
        std::vector<Eigen::Matrix3d> A_wp_feet(NF);

        for (int i = 0; i < NF; ++i)
        {
            Eigen::Matrix3d sf  = utils::skew(u.segment<3>(_O_F + 3 * i));
            dtau_dp_com        += sf;
            A_wp_feet[i]        = -_params.dt * (I_w_inv * sf);
        }
        Eigen::Matrix3d A_wp_com = _params.dt * (I_w_inv * dtau_dp_com);

        // Assemble dX
        dX.setZero(_params.ndq, _params.ndq);
        dX.block<3, 3>(_O_P_COM, _O_P_COM) = Eigen::Matrix3d::Identity();
        dX.block<3, 3>(_O_P_COM, _O_V_COM) =
          _params.dt * Eigen::Matrix3d::Identity();

        dX.block<3, 3>(_O_THETA, _O_P_COM) = _params.dt * (J_r * A_wp_com);
        dX.block<3, 3>(_O_THETA, _O_THETA) =
          R_dq.transpose() + _params.dt * (J_r * A_wtheta);
        dX.block<3, 3>(_O_THETA, _O_OMEGA) = _params.dt * (J_r * A_ww);
        for (int i = 0; i < NF; ++i)
        {
            dX.block<3, 3>(_O_THETA, _O_P_FEET + 3 * i) =
              _params.dt * (J_r * A_wp_feet[i]);
        }

        dX.block<3, 3>(_O_V_COM, _O_V_COM) = Eigen::Matrix3d::Identity();

        dX.block<3, 3>(_O_OMEGA, _O_P_COM) = A_wp_com;
        dX.block<3, 3>(_O_OMEGA, _O_THETA) = A_wtheta;
        dX.block<3, 3>(_O_OMEGA, _O_OMEGA) = A_ww;
        for (int i = 0; i < NF; ++i)
        {
            dX.block<3, 3>(_O_OMEGA, _O_P_FEET + 3 * i) = A_wp_feet[i];
            dX.block<3, 3>(_O_P_FEET + 3 * i, _O_P_FEET + 3 * i) =
              Eigen::Matrix3d::Identity();
        }

        // Assemble dU
        dU.setZero(_params.ndq, _params.nu);
        for (int i = 0; i < NF; ++i)
        {
            int             cf   = _O_F + 3 * i;
            int             cv   = _O_V + 3 * i;
            Eigen::Matrix3d S_ri = utils::skew(r[i]);

            dU.block<3, 3>(_O_V_COM, cf) =
              (_params.dt / _params.mass) * Eigen::Matrix3d::Identity();
            dU.block<3, 3>(_O_P_COM, cf) =
              ((_params.dt * _params.dt) / _params.mass) *
              Eigen::Matrix3d::Identity();

            Eigen::Matrix3d dU_wi        = _params.dt * (I_w_inv * S_ri);
            dU.block<3, 3>(_O_OMEGA, cf) = dU_wi;
            dU.block<3, 3>(_O_THETA, cf) = _params.dt * (J_r * dU_wi);
            dU.block<3, 3>(_O_P_FEET + 3 * i, cv) =
              _params.dt * Eigen::Matrix3d::Identity();
        }
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxx) const override
    {
        Pxx.setZero();
    }
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxu) const override
    {
        Pxu.setZero();
    }
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Puu) const override
    {
        Puu.setZero();
    }
};

// ---------------------------------------------------------------------------
// Context Reference Structure (Shared between Costs/Constraints)
// ---------------------------------------------------------------------------
struct SRBReferences
{
    Eigen::Vector3d              com_ref;
    Eigen::Vector4d              q_ref;
    std::vector<Eigen::Vector3d> p_foot_ref;
    std::vector<bool>            is_swing;
    bool                         terminal = false;

    SRBReferences() : p_foot_ref(NF), is_swing(NF, false) {}
};

// ---------------------------------------------------------------------------
// Splitting Costs
// ---------------------------------------------------------------------------

class CoMTrackingCost : public StageCost
{
    SRBDParameters _params;
    SRBReferences *_refs;

  public:
    CoMTrackingCost(const SRBDParameters &params, SRBReferences *refs)
      : _params(params)
      , _refs(refs)
    {
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        Eigen::Vector3d err = x.segment<3>(_O_P_COM) - _refs->com_ref;
        return (Scalar() << _params.w_p * err.squaredNorm()).finished();
    }

    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        dx.setZero();
        du.setZero();
        dx.segment<3>(_O_P_COM) =
          2.0 * _params.w_p * (x.segment<3>(_O_P_COM) - _refs->com_ref);
    }

    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        Hxx.setZero();
        Hxx.block<3, 3>(_O_P_COM, _O_P_COM) =
          2.0 * _params.w_p * Eigen::Matrix3d::Identity();
    }
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        Hxu.setZero();
    }
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        Huu.setZero();
    }
};

class OrientationTrackingCost : public StageCost
{
    SRBDParameters _params;
    SRBReferences *_refs;

  public:
    OrientationTrackingCost(const SRBDParameters &params, SRBReferences *refs)
      : _params(params)
      , _refs(refs)
    {
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        Eigen::Vector4d q_ref_inv(-_refs->q_ref(0),
                                  -_refs->q_ref(1),
                                  -_refs->q_ref(2),
                                  _refs->q_ref(3));
        Eigen::Vector4d q_err =
          utils::quat_mul(q_ref_inv, x.segment<4>(_O_THETA));
        return (Scalar() << _params.w_o * q_err.segment<3>(0).squaredNorm())
          .finished();
    }

    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        dx.setZero();
        du.setZero();
        Eigen::Vector4d q_ref_inv(-_refs->q_ref(0),
                                  -_refs->q_ref(1),
                                  -_refs->q_ref(2),
                                  _refs->q_ref(3));
        Eigen::Vector4d q_err =
          utils::quat_mul(q_ref_inv, x.segment<4>(_O_THETA));
        dx.segment<3>(_O_THETA) = 2.0 * _params.w_o * q_err.segment<3>(0);
    }

    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        Hxx.setZero();
        Hxx.block<3, 3>(_O_THETA, _O_THETA) =
          2.0 * _params.w_o * Eigen::Matrix3d::Identity();
    }
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        Hxu.setZero();
    }
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        Huu.setZero();
    }
};

class AngularVelocityCost : public StageCost
{
    SRBDParameters _params;

  public:
    AngularVelocityCost(const SRBDParameters &params) : _params(params) {}

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        return (Scalar() << _params.w_w * x.segment<3>(_O_OMEGA).squaredNorm())
          .finished();
    }

    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        dx.setZero();
        du.setZero();
        dx.segment<3>(_O_OMEGA) = 2.0 * _params.w_w * x.segment<3>(_O_OMEGA);
    }

    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        Hxx.setZero();
        Hxx.block<3, 3>(_O_OMEGA, _O_OMEGA) =
          2.0 * _params.w_w * Eigen::Matrix3d::Identity();
    }
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        Hxu.setZero();
    }
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        Huu.setZero();
    }
};

class FootTrackingCost : public StageCost
{
    SRBDParameters _params;
    SRBReferences *_refs;

  public:
    FootTrackingCost(const SRBDParameters &params, SRBReferences *refs)
      : _params(params)
      , _refs(refs)
    {
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        double cost = 0;
        for (int i = 0; i < NF; ++i)
        {
            cost += _params.w_pf *
                    (x.segment<3>(_O_P_FEET + 3 * i) - _refs->p_foot_ref[i])
                      .squaredNorm();
        }
        return (Scalar() << cost).finished();
    }

    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        dx.setZero();
        du.setZero();
        for (int i = 0; i < NF; ++i)
        {
            dx.segment<3>(_O_P_FEET + 3 * i) =
              2.0 * _params.w_pf *
              (x.segment<3>(_O_P_FEET + 3 * i) - _refs->p_foot_ref[i]);
        }
    }

    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        Hxx.setZero();
        for (int i = 0; i < NF; ++i)
        {
            Hxx.block<3, 3>(_O_P_FEET + 3 * i, _O_P_FEET + 3 * i) =
              2.0 * _params.w_pf * Eigen::Matrix3d::Identity();
        }
    }
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        Hxu.setZero();
    }
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        Huu.setZero();
    }
};

class ControlRegularizationCost : public StageCost
{
    SRBDParameters _params;
    SRBReferences *_refs;

  public:
    ControlRegularizationCost(const SRBDParameters &params, SRBReferences *refs)
      : _params(params)
      , _refs(refs)
    {
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        if (_refs->terminal) return Scalar::Zero();
        double cost = 0;
        for (int i = 0; i < NF; ++i)
        {
            if (!_refs->is_swing[i])
            {
                cost += _params.w_f * u.segment<3>(_O_F + 3 * i).squaredNorm();
            }
            cost += _params.w_vf * u.segment<3>(_O_V + 3 * i).squaredNorm();
        }
        return (Scalar() << cost).finished();
    }

    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        dx.setZero();
        du.setZero();
        if (_refs->terminal) return;
        for (int i = 0; i < NF; ++i)
        {
            if (!_refs->is_swing[i])
            {
                du.segment<3>(_O_F + 3 * i) =
                  2.0 * _params.w_f * u.segment<3>(_O_F + 3 * i);
            }
            du.segment<3>(_O_V + 3 * i) =
              2.0 * _params.w_vf * u.segment<3>(_O_V + 3 * i);
        }
    }

    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        Hxx.setZero();
    }
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        Hxu.setZero();
    }
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        Huu.setZero();
        if (_refs->terminal) return;
        for (int i = 0; i < NF; ++i)
        {
            if (!_refs->is_swing[i])
            {
                Huu.block<3, 3>(_O_F + 3 * i, _O_F + 3 * i) =
                  2.0 * _params.w_f * Eigen::Matrix3d::Identity();
            }
            Huu.block<3, 3>(_O_V + 3 * i, _O_V + 3 * i) =
              2.0 * _params.w_vf * Eigen::Matrix3d::Identity();
        }
    }
};

// ---------------------------------------------------------------------------
// Splitting Constraints
// ---------------------------------------------------------------------------

class CoMHeightConstraint : public StageConstraints
{
  public:
    uint output_size() const override
    {
        return 1;
    }
    const VecX &lower_bound() const override
    {
        return _lb;
    }
    const VecX &upper_bound() const override
    {
        return _ub;
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        return x.segment<1>(_O_P_COM + 2); // z
    }
    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        dX.setZero();
        dU.setZero();
        dX(0, _O_P_COM + 2) = 1.0;
    }
    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override
    {
        Pxx.setZero();
    }
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override
    {
        Pxu.setZero();
    }
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override
    {
        Puu.setZero();
    }

  private:
    const VecX _lb = VecX::Constant(1, Z_MIN);
    const VecX _ub = VecX::Constant(1, std::numeric_limits<double>::infinity());
};

class FrictionAndForcesConstraint : public StageConstraints
{
    SRBReferences *_refs;
    int            _foot_idx;
    mutable VecX   _lb_cache, _ub_cache;

  public:
    FrictionAndForcesConstraint(SRBReferences *refs, int foot_idx)
      : _refs(refs)
      , _foot_idx(foot_idx)
    {
    }

    uint output_size() const override
    {
        return 5;
    }

    // is_swing[_foot_idx] changes between solves, so the bound must be
    // recomputed on every call rather than cached at construction; the
    // mutable members below just give lower_bound()/upper_bound() somewhere
    // stable to return a reference to
    const VecX &lower_bound() const override
    {
        _lb_cache.setZero(5);
        if (!_refs->is_swing[_foot_idx])
        {
            _lb_cache(4) = F_Z_MIN;
        }
        return _lb_cache;
    }

    const VecX &upper_bound() const override
    {
        _ub_cache.setZero(5);
        if (!_refs->is_swing[_foot_idx])
        {
            _ub_cache.setConstant(std::numeric_limits<double>::infinity());
            _ub_cache(4) = F_Z_MAX;
        }
        return _ub_cache;
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        VecX   c(5);
        double fx = u(_O_F + 3 * _foot_idx);
        double fy = u(_O_F + 3 * _foot_idx + 1);
        double fz = u(_O_F + 3 * _foot_idx + 2);
        c(0)      = fx + MU * fz;
        c(1)      = -fx + MU * fz;
        c(2)      = fy + MU * fz;
        c(3)      = -fy + MU * fz;
        c(4)      = fz;
        return c;
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        dX.setZero();
        dU.setZero();
        int cf        = _O_F + 3 * _foot_idx;
        dU(0, cf)     = 1.0;
        dU(0, cf + 2) = MU;
        dU(1, cf)     = -1.0;
        dU(1, cf + 2) = MU;
        dU(2, cf + 1) = 1.0;
        dU(2, cf + 2) = MU;
        dU(3, cf + 1) = -1.0;
        dU(3, cf + 2) = MU;
        dU(4, cf + 2) = 1.0;
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override
    {
        Pxx.setZero();
    }
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override
    {
        Pxu.setZero();
    }
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override
    {
        Puu.setZero();
    }
};

class FootVelocityConstraint : public StageConstraints
{
    SRBReferences *_refs;
    int            _foot_idx;
    mutable VecX   _lb_cache, _ub_cache;

  public:
    FootVelocityConstraint(SRBReferences *refs, int foot_idx)
      : _refs(refs)
      , _foot_idx(foot_idx)
    {
    }

    uint output_size() const override
    {
        return 3;
    }

    // is_swing[_foot_idx] changes between solves, so the bound must be
    // recomputed on every call; see the comment in FrictionAndForcesConstraint
    // above
    const VecX &lower_bound() const override
    {
        if (_refs->is_swing[_foot_idx])
        {
            _lb_cache.setConstant(3, -std::numeric_limits<double>::infinity());
        }
        else
        {
            _lb_cache.setZero(3);
        }
        return _lb_cache;
    }

    const VecX &upper_bound() const override
    {
        if (_refs->is_swing[_foot_idx])
        {
            _ub_cache.setConstant(3, std::numeric_limits<double>::infinity());
        }
        else
        {
            _ub_cache.setZero(3);
        }
        return _ub_cache;
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        return u.segment<3>(_O_V + 3 * _foot_idx);
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        dX.setZero();
        dU.setZero();
        dU.block<3, 3>(0, _O_V + 3 * _foot_idx) = Eigen::Matrix3d::Identity();
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override
    {
        Pxx.setZero();
    }
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override
    {
        Pxu.setZero();
    }
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override
    {
        Puu.setZero();
    }
};