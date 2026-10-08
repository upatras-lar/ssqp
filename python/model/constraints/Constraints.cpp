#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
#include "../model_reference.hpp"
#include "model/stagewise/StageConstraints.hpp"
#include "model/stagewise/predefined/Constraints.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model;
using namespace model::stagewise;
using namespace model::stagewise::predefined::constraints;

struct __attribute__((visibility("default"))) PyStageConstraints
  : StageConstraints
{
    NB_TRAMPOLINE(StageConstraints, 10);

    uint output_size() const override
    {
        try
        {
            NB_OVERRIDE_PURE(output_size);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageConstraints::output_size`: expected an integer");
        }
    }

    // Not using NB_OVERRIDE here: it deduces its return type from
    // StageConstraints::lower_bound/upper_bound, which now return
    // `const VecX&`, and casting a Python return value straight into a
    // reference would leave it dangling
    // Cast into an owned VecX instead and cache it, so we can hand back a
    // reference with a stable lifetime
    const VecX &lower_bound() const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "lower_bound",
                                               false);
            _lb_cache =
              nb_ticket.key.is_valid()
                ? nb::cast<VecX>(nb_trampoline.base().attr(nb_ticket.key)())
                : NBBase::lower_bound();

            return _lb_cache;
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageConstraints::lower_bound`: "
                                     "expected a vector (numpy.ndarray)");
        }
    }

    const VecX &upper_bound() const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "upper_bound",
                                               false);
            _ub_cache =
              nb_ticket.key.is_valid()
                ? nb::cast<VecX>(nb_trampoline.base().attr(nb_ticket.key)())
                : NBBase::upper_bound();

            return _ub_cache;
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageConstraints::upper_bound`: "
                                     "expected a vector (numpy.ndarray)");
        }
    }

    uint constraint_diff_size() const override
    {
        try
        {
            NB_OVERRIDE(constraint_diff_size);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageConstraints::constraint_diff_size`: expected an integer");
        }
    }

    VecX right_minus(const VecX &x, const VecX &y) const override
    {
        try
        {
            NB_OVERRIDE(right_minus, x, y);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageConstraints::right_minus`: "
                                     "expected a vector (numpy.ndarray)");
        }
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline, "evaluate", true);
            return nb::cast<VecX>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageConstraints::evaluate`: "
                                     "expected a vector (numpy.ndarray)");
        }
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline, "gradient", true);
            std::tie(dX, dU) = nb::cast<std::pair<MatX, MatX>>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageConstraints::gradient`: "
              "expected a pair of matrices (numpy.ndarray, numpy.ndarray)");
        }
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessprod_xx",
                                               true);
            Pxx = nb::cast<MatX>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u, mu));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageConstraints::hessprod_xx`: "
                                     "expected a matrix (numpy.ndarray)");
        }
    }

    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessprod_xu",
                                               true);
            Pxu = nb::cast<MatX>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u, mu));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageConstraints::hessprod_xu`: "
                                     "expected a matrix (numpy.ndarray)");
        }
    }

    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessprod_uu",
                                               true);
            Puu = nb::cast<MatX>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u, mu));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageConstraints::hessprod_uu`: "
                                     "expected a matrix (numpy.ndarray)");
        }
    }

  private:
    // Backing storage for lower_bound()/upper_bound(); see the comment above
    // those overrides
    mutable VecX _lb_cache, _ub_cache;
};

void bind_Constraints(nb::module_ &m)
{
    nb::class_<StageConstraints, PyStageConstraints>(m,
                                                     "StageConstraints",
                                                     R"doc(
        General abstract constraint function for an optimization stage $k$: $\boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
        Defines the required functions for stage constraints, while providing 
        default implementations for some
        )doc",
                                                     nb::dynamic_attr())
      .def(nb::init())
      .def(nb::init<const std::string &>(), nb::arg("name") = "")
      .def("names",
           &StageConstraints::names,
           "Get all constraints' names for StageConstraints.")
      .def("set_enabled",
           &StageConstraints::set_enabled,
           nb::arg("enabled"),
           "Set enabled status for StageConstraints")
      .def("enabled",
           &StageConstraints::enabled,
           "Whether this constraint is enabled")
      .def("output_size",
           &StageConstraints::output_size,
           "Get the constraints function output size.")
      .def("lower_bound",
           &StageConstraints::lower_bound,
           "Get the constraints' lower bound vector.")
      .def("upper_bound",
           &StageConstraints::upper_bound,
           "Get the constraints' upper bound vector.")
      .def("constraint_diff_size",
           &StageConstraints::constraint_diff_size,
           R"doc(
        Size of the difference vector: $\boldsymbol{c}_k(\boldsymbol{x}_1, \boldsymbol{u}_1) \ominus \boldsymbol{c}_k(\boldsymbol{x}_2, \boldsymbol{u}_2)$
        Where $\ominus$: the defined right_minus operator
           )doc")
      .def("right_minus",
           &StageConstraints::right_minus,
           nb::arg("x"),
           nb::arg("y"),
           R"doc(
        The $\ominus$ operator, which outputs the difference of two constraint
        vectors in tangent space
            )doc")
      .def("evaluate",
           &StageConstraints::evaluate,
           nb::arg("x"),
           nb::arg("u"),
           R"doc(
        Evaluate function $f(\boldsymbol{x}, \boldsymbol{u})$
        Returns: output vector, $n \times 1$
            )doc")
      .def(
        "gradient",
        [](const StageConstraints &self, const VecX &x, const VecX &u)
        {
            MatX dX, dU;
            self.gradient(x, u, dX, dU);
            return nb::make_tuple(dX, dU);
        },
        nb::arg("x"),
        nb::arg("u"),
        R"doc(
        Get constraint Jacobians with respect to $\boldsymbol{x}$ and $\boldsymbol{u}$, evaluated at
        given $(\boldsymbol{x}, \boldsymbol{u})$: $\nabla_x \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$ and $\nabla_u \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         dX: gradient with respect to $\boldsymbol{x}$, $n_c \times n_{dx}$
         dU: gradient with respect to $\boldsymbol{u}$, $n_c \times n_u$
        )doc")
      .def(
        "hessprod_xx",
        [](const StageConstraints &self,
           const VecX &x,
           const VecX &u,
           const VecX &mu)
        {
            MatX Pxx;
            self.hessprod_xx(x, u, mu, Pxx);
            return Pxx;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("mu"),
        R"doc(
        Get product $\nabla_{x} \mu_k^\top \nabla_{x} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Pxx: Product, $n_{dx} \times n_{dx}$
        )doc")
      .def(
        "hessprod_xu",
        [](const StageConstraints &self,
           const VecX &x,
           const VecX &u,
           const VecX &mu)
        {
            MatX Pxu;
            self.hessprod_xu(x, u, mu, Pxu);
            return Pxu;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("mu"),
        R"doc(
        Get product $\nabla_{x} \mu_k^\top \nabla_{u} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Pxu: Product, $n_{dx} \times n_u$
        )doc")
      .def(
        "hessprod_uu",
        [](const StageConstraints &self,
           const VecX &x,
           const VecX &u,
           const VecX &mu)
        {
            MatX Puu;
            self.hessprod_uu(x, u, mu, Puu);
            return Puu;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("mu"),
        R"doc(
        Get product $\nabla_{u} \mu_k^\top \nabla_{u} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Puu: Product, $n_u \times n_u$
        )doc");

    nb::class_<JointAngleConstraint, StageConstraints>(m,
                                                       "JointAngleConstraint",
                                                       R"doc(
        Predefined joint angle constraint

        Creates a constraint on the joint angles of the model

        Note: this constraint is separate to any potential innate constraints
        $\boldsymbol{q}_l \leq \boldsymbol{q} \leq \boldsymbol{q}_u$

        where:
         - $\boldsymbol{q}_l$, $\boldsymbol{q}_u$: Lower and upper configuration vectors
         - $\boldsymbol{q}$: The model's joint configuration vector

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`), appearing after the floating base (if present)
           )doc")
      .def(nb::init<const Model &,
                    const VecX &,
                    const VecX &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &JointAngleConstraint::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &JointAngleConstraint::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<JointVelocityConstraint, StageConstraints>(
      m,
      "JointVelocityConstraint",
      R"doc(
        Predefined joint velocity constraint

        Creates a constraint on the joint velocities of the model

        Note: this constraint is separate to any potential innate constraints
        $\boldsymbol{q}_{v,l} \leq \boldsymbol{q}_v \leq \boldsymbol{q}_{v, u}$

        where:
         - $\boldsymbol{q}_{v,l}$, $\boldsymbol{q}_{v,u}$: Lower and upper joint velocity vectors
         - $\boldsymbol{q}_v$: The model's joint velocity vector

        Assumes:
         - The model joint velocity is the final part of the state vector (size
           `model.nv()`)
           )doc")
      .def(nb::init<const Model &,
                    const VecX &,
                    const VecX &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &JointVelocityConstraint::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &JointVelocityConstraint::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<JointControlConstraint, StageConstraints>(
      m,
      "JointControlConstraint",
      R"doc(
        Predefined joint control constraint

        Creates a constraint on the joint part of the control vector

        This can either act as a torque constraint (FD mode) or an acceleration
        constraint (ID mode)

        Note: this constraint is separate to any potential innate constraints
        $\boldsymbol{u}_{c,l} \leq \boldsymbol{u}_c \leq \boldsymbol{u}_{c, u}$

        where:
         - $\boldsymbol{u}_{c,l}$, $\boldsymbol{u}_{c,u}$: Lower and upper joint control vectors
         - $\boldsymbol{u}_c$: The joint part of the control vector

        Assumes:
         - The model control is the first part of the control vector (size 
           `model.nu()`)
           )doc")
      .def(nb::init<const Model &,
                    const VecX &,
                    const VecX &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &JointControlConstraint::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &JointControlConstraint::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FloatingBaseTranslationBound, StageConstraints>(
      m,
      "FloatingBaseTranslationBound",
      R"doc(
        Floating base position bound constraint

        Creates a linear bound constraint on the position of the floating base:
        $\boldsymbol{b}_l \leq p_f \leq \boldsymbol{b}_u$

        where:
        - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the base position (in $\mathcal{R}^3$)
        - $p_f$: Floating base position extracted from the system state

        Assumes:
        - The Model contains a floating-base where the base position is the first
        part of the configuration vector (first 3 elements of size `model.nq()`)
        )doc")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &FloatingBaseTranslationBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FloatingBaseTranslationBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FloatingBaseRotationBound, StageConstraints>(
      m,
      "FloatingBaseRotationBound",
      R"doc(
        Predefined floating base orientation bound

        Creates a nonlinear bound constraint on the orientation of the floating base:
        $\boldsymbol{b}_l \leq \log_{SO3}(R_{ref}^\top R_f) \leq \boldsymbol{b}_u$

        where:
        - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the orientation error (in $\mathcal{R}^3$)
        - $R_f$: Floating base rotation matrix
        - $R_{ref}$: Reference rotation matrix

        Assumes:
        - The model joint configuration is the first part of the state vector (size
        `model.nq()`)
        - The floating base orientation is represented as a rotation matrix in SE(3)
        and can be extracted from the base pose in the system state
        )doc")
      .def(nb::init<const Model &,
                    const Mat3 &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("Rref"),
           nb::arg("lb")   = Vec3::Zero(),
           nb::arg("ub")   = Vec3::Zero(),
           nb::arg("name") = "")
      .def("set_Rref", &FloatingBaseRotationBound::set_Rref, nb::arg("Rref"))
      .def("set_lower_bound",
           &FloatingBaseRotationBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FloatingBaseRotationBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FloatingBasePoseBound, StageConstraints>(m,
                                                        "FloatingBasePoseBound",
                                                        R"doc(
        Predefined floating base SE(3) pose bound

        Creates a nonlinear bound constraint on the full pose (position + orientation)
        of the floating base:
        $\boldsymbol{b}_l \leq \log_{SE3}(T_{ref}^{-1} T_f) \leq \boldsymbol{b}_u$

        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the pose error (in $\mathcal{R}^6$)
         - $T_f$: Floating base pose in SE(3)
         - $T_{ref}$: Reference pose in SE(3)

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)
         - The floating base pose is represented as an SE(3) transform and can be
           extracted from the base state in the system model
        )doc")
      .def(nb::init<const Model &,
                    const Mat4 &,
                    const Vec6 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("Tref"),
           nb::arg("lb")   = Vec6::Zero(),
           nb::arg("ub")   = Vec6::Zero(),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Mat3 &,
                    const Vec6 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("tref"),
           nb::arg("Rref"),
           nb::arg("lb")   = Vec6::Zero(),
           nb::arg("ub")   = Vec6::Zero(),
           nb::arg("name") = "")
      .def("set_Tref", &FloatingBasePoseBound::set_Tref, nb::arg("Tref"))
      .def("set_tref", &FloatingBasePoseBound::set_tref, nb::arg("tref"))
      .def("set_Rref", &FloatingBasePoseBound::set_Rref, nb::arg("Rref"))
      .def("set_lower_bound",
           &FloatingBasePoseBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FloatingBasePoseBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FloatingBaseLinearVelocityBound, StageConstraints>(
      m,
      "FloatingBaseLinearVelocityBound",
      R"doc(
        Predefined floating base linear velocity bound

        Creates a linear bound constraint on the linear velocity of the floating base:
        $\boldsymbol{b}_l \leq \boldsymbol{v}_b \leq \boldsymbol{b}_u$

        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
         - $\boldsymbol{v}_b$: Floating base linear velocity vector (in $\mathcal{R}^3$)

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)
         - The model joint velocity is the final part of the state vector (size
           `model.nv()`)
           )doc")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &FloatingBaseLinearVelocityBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FloatingBaseLinearVelocityBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FloatingBaseAngularVelocityBound, StageConstraints>(
      m,
      "FloatingBaseAngularVelocityBound",
      R"doc(
        Predefined floating base angular velocity bound

        Creates a linear bound constraint on the angular velocity of the floating base:
        $\boldsymbol{b}_l \leq \boldsymbol{\omega}_b \leq \boldsymbol{b}_u$

        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
         - $\boldsymbol{\omega}_b$: Floating base angular velocity vector (in $\mathcal{R}^3$)

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)
         - The model joint velocity is the final part of the state vector (size
           `model.nv()`)
           )doc")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &FloatingBaseAngularVelocityBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FloatingBaseAngularVelocityBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FrameTranslationBound, StageConstraints>(m,
                                                        "FrameTranslationBound",
                                                        R"doc(
        Predefined 3D position bound for a given frame

        Creates a linear bound constraint on the position of a frame of the given
        model: $\boldsymbol{b}_l \leq \boldsymbol{p}_f \leq \boldsymbol{b}_u$

        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
         - $\boldsymbol{p}_f$: Frame position vector (in $\mathcal{R^3}$)

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
           )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &FrameTranslationBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FrameTranslationBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FrameTerrainHeightBound, StageConstraints>(
      m,
      "FrameTerrainHeightBound",
      R"doc(
        Predefined terrain height bound for a given frame

        Creates a bound constraint on the height of a frame above the model's
        terrain: $b_l \leq p_z - h(p_x, p_y) \leq b_u$

        where:
         - $b_l$, $b_u$: Lower and upper bounds on the height above the terrain
         - $\boldsymbol{p}_f$: Frame position vector (in $\mathcal{R^3}$)
         - $h$: Terrain height at the frame's horizontal position

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)
           )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const double,
                    const double,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &FrameTerrainHeightBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FrameTerrainHeightBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FrameRotationBound, StageConstraints>(m,
                                                     "FrameRotationBound",
                                                     R"doc(
        Predefined 3D orientation bound for a given frame

        Creates a nonlinear bound constraint on the orientation of a frame:
        $\boldsymbol{b}_l \leq \log_{SO3}R_{ref}^\top R_f \leq \boldsymbol{b}_u$

        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the orientation error (in $\mathcal{R}^3$)
         - $R_f$: Frame rotation matrix
         - $R_{ref}$: Reference rotation matrix

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
           )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Mat3 &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("Rref"),
           nb::arg("lb")   = Vec3::Zero(),
           nb::arg("ub")   = Vec3::Zero(),
           nb::arg("name") = "")
      .def("set_Rref", &FrameRotationBound::set_Rref, nb::arg("Rref"))
      .def("set_lower_bound",
           &FrameRotationBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FrameRotationBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FramePoseBound, StageConstraints>(m, "FramePoseBound", R"doc(
        Predefined SE(3) pose bound for a given frame

        Creates a nonlinear bound constraint on the pose of a frame:
        $\boldsymbol{b}_l \leq \log_{SE3}(T_{ref}^{-1} T_f) \leq \boldsymbol{b}_u$

        where:
        - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the pose error (in $\mathcal{R}^6$)
        - $T_f$: Frame pose in SE(3)
        - $T_{ref}$: Reference pose in SE(3)

        Assumes:
        - The model joint configuration is the first part of the state vector (size
        `model.nq()`)
        )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Mat4 &,
                    const Vec6 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("Tref"),
           nb::arg("lb")   = Vec6::Zero(),
           nb::arg("ub")   = Vec6::Zero(),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Mat3 &,
                    const Vec6 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("tref"),
           nb::arg("Rref"),
           nb::arg("lb")   = Vec6::Zero(),
           nb::arg("ub")   = Vec6::Zero(),
           nb::arg("name") = "")
      .def("set_Tref", &FramePoseBound::set_Tref, nb::arg("Tref"))
      .def("set_tref", &FramePoseBound::set_tref, nb::arg("tref"))
      .def("set_Rref", &FramePoseBound::set_Rref, nb::arg("Rref"))
      .def("set_lower_bound",
           &FramePoseBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FramePoseBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FrameLinearVelocityBound, StageConstraints>(
      m,
      "FrameLinearVelocityBound",
      R"doc(
        Predefined 3D velocity bound for a given frame

        Creates a linear bound constraint on the velocity of a frame of the given
        model: $\boldsymbol{b}_l \leq \boldsymbol{v}_f \leq \boldsymbol{b}_u$
        
        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
         - $\boldsymbol{v}_f$: Frame velocity vector (in $\mathcal{R^3}$)

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
         - The model joint velocity is the final part of the state vector (size
           `model.nv()`)
           )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &FrameLinearVelocityBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FrameLinearVelocityBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FrameAngularVelocityBound, StageConstraints>(
      m,
      "FrameAngularVelocityBound",
      R"doc(
        Predefined angular velocity bound for a given frame

        Creates a linear bound constraint on the angular velocity of a frame of the
        given model:
        $\boldsymbol{b}_l \leq \boldsymbol{\omega}_f \leq \boldsymbol{b}_u$

        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
         - $\boldsymbol{\omega}_f$: Frame angular velocity vector (in $\mathcal{R}^3$)

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)
         - The model joint velocity is the final part of the state vector (size
           `model.nv()`)

           )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &FrameAngularVelocityBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FrameAngularVelocityBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<CenterOfMassBound, StageConstraints>(m,
                                                    "CenterOfMassBound",
                                                    R"doc(
        Predefined center of mass bound

        Creates a linear bound constraint on the center of mass of the model:
        $\boldsymbol{b}_l \leq \boldsymbol{c} \leq \boldsymbol{b}_u$

        where:
        - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
        - $\boldsymbol{c}$ is the center of mass of the model

        Assumes:
        - The model joint configuration is the first part of the state vector (size 
        `model.nq()`)
        )doc")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::call_policy<keep_model>(),
           nb::arg("model"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("name") = "")
      .def("set_lower_bound",
           &CenterOfMassBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &CenterOfMassBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<ContactConstraint, StageConstraints>(m,
                                                    "ContactConstraint",
                                                    R"doc(
        Predefined contact constraint for a given frame
        Creates a set of constraints that amount to a contact force constraint for
        a frame
        This consists of the following 4-point linearized friction cone constraints:
         - $f_n \gt 0$
         - $|f_1| \leq \mu f_n$
         - $|f_2| \leq \mu f_n$
        As well as the following constraint on the 3D position of the frame the
        contact is on:
         - $p_z = h_z$
        And, unless `no_slip` is false, the following no-slip constraint on the
        linear velocity of the frame:
         - $\boldsymbol{v}_f = \boldsymbol{0}$
        where:
         - $f_n$ is the normal of the contact force ($f_z$)
         - $f_1$ is the first component of the tangential contact force ($f_x$)
         - $f_2$ is the second component of the tangential contact force ($f_y$)
         - $\mu$ is the static friction coefficient at the contact
         - $p_z$ is the $z-$component of the translation of the frame the contact is on
         - $h_z$ is the height of the terrain at point $(p_x, p_y)$
         - $\boldsymbol{v}_f$ is the linear velocity of the frame the contact is on
        Note that this constraint cannot be manually added in the Python library,
        use the `Model`'s `addContactConstraint` method instead.
        )doc")
      .def("set_mu", &ContactConstraint::set_mu, nb::arg("mu"))
      .def("set_max_normal_force",
           &ContactConstraint::set_max_normal_force,
           nb::arg("max_normal_force"))
      .def("set_eq_tol", &ContactConstraint::set_eq_tol, nb::arg("eq_tol"));
}
}
}
