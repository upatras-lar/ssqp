#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
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
    NB_TRAMPOLINE(StageConstraints, 12);

    void set_enabled(const bool enabled) override
    {
        NB_OVERRIDE(set_enabled, enabled);
    }

    bool enabled() const override
    {
        NB_OVERRIDE(enabled);
    }

    uint output_size() const override
    {
        NB_OVERRIDE_PURE(output_size);
    }

    VecX lower_bound() const override
    {
        NB_OVERRIDE(lower_bound);
    }

    VecX upper_bound() const override
    {
        NB_OVERRIDE(upper_bound);
    }

    uint constraint_diff_size() const override
    {
        NB_OVERRIDE(constraint_diff_size);
    }

    VecX right_minus(const VecX &x, const VecX &y) const override
    {
        NB_OVERRIDE(right_minus, x, y);
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "evaluate", true);
        return nb::cast<VecX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "gradient", true);
        std::tie(dX, dU) = nb::cast<std::pair<MatX, MatX>>(
          nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxx) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessprod_xx", true);
        Pxx = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxu) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessprod_xu", true);
        Pxu = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Puu) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessprod_uu", true);
        Puu = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }
};

void bind_Constraints(nb::module_ &m)
{
    nb::class_<StageConstraints, PyStageConstraints>(m,
                                                     "StageConstraints",
                                                     R"doc(
        General abstract constraint function for an optimization stage $k$: $\boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
        Defines the required functions for stage constraints, while providing 
        default implementations for some
        )doc")
      .def(nb::init())
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
           const VecX             &x,
           const VecX             &u,
           const VecX             &mu)
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
           const VecX             &x,
           const VecX             &u,
           const VecX             &mu)
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
           const VecX             &x,
           const VecX             &u,
           const VecX             &mu)
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
                                                       "JointAngleConstraint")
      .def(nb::init<const VecX &,
                    const VecX &,
                    const uint,
                    const uint,
                    const bool>(),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("free_flyer") = false,
           R"doc(
        Predefined joint angle constraint
        Creates a constraint on the joint angles of the model
        Note: this constraint is separate to any potential innate constraints
        $\boldsymbol{q}_l \leq \boldsymbol{q} \leq \boldsymbol{q}_u$
        where:
         - $\boldsymbol{q}_l$, $\boldsymbol{q}_u$: Lower and upper configuration vectors
         - $\boldsymbol{q}$: The model's joint configuration vector
           )doc")
      .def("set_lower_bound",
           &JointAngleConstraint::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &JointAngleConstraint::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<JointVelocityConstraint, StageConstraints>(
      m,
      "JointVelocityConstraint")
      .def(nb::init<const VecX &, const VecX &, const uint, const bool>(),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nv"),
           nb::arg("free_flyer") = false,
           R"doc(
        Predefined joint velocity constraint
        Creates a constraint on the joint velocities of the model
        Note: this constraint is separate to any potential innate constraints
        $\boldsymbol{q}_{v,l} \leq \boldsymbol{q}_v \leq \boldsymbol{q}_{v, u}$
        where:
         - $\boldsymbol{q}_{v,l}$, $\boldsymbol{q}_{v,u}$: Lower and upper joint velocity vectors
         - $\boldsymbol{q}_v$: The model's joint velocity vector
           )doc")
      .def("set_lower_bound",
           &JointVelocityConstraint::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &JointVelocityConstraint::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<JointControlConstraint, StageConstraints>(
      m,
      "JointControlConstraint")
      .def(nb::init<const VecX &,
                    const VecX &,
                    const uint,
                    const uint,
                    const bool>(),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nv"),
           nb::arg("nu"),
           nb::arg("free_flyer") = false,
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
           )doc")
      .def("set_lower_bound",
           &JointControlConstraint::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &JointControlConstraint::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FramePositionBound, StageConstraints>(m, "FramePositionBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Vec3 &,
                    uint,
                    uint>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nq"),
           nb::arg("nv"),
           R"doc(
        Predefined 3D position bound for a given frame
        Creates a linear bound constraint on the position of a frame of the given
        model: $\boldsymbol{b}_l <= \boldsymbol{p}_f <= \boldsymbol{b}_u$
        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
         - $\boldsymbol{p}_f$: Frame position vector (in $\mathcal{R^3}$)
           )doc")
      .def("set_lower_bound",
           &FramePositionBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FramePositionBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FrameRotationBound, StageConstraints>(m, "FrameRotationBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Mat3 &,
                    const Vec3 &,
                    const Vec3 &,
                    uint,
                    uint>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("Rref"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nq"),
           nb::arg("nv"),
           R"doc(
        Predefined 3D orientation bound for a given frame
        Creates a nonlinear bound constraint on the orientation of a frame:
        $\boldsymbol{b}_l \leq \log_{SO3}R_{ref}^\top R_f \leq \boldsymbol{b}_u$
        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the orientation error (in $\mathcal{R}^3$)
         - $R_f$: Frame rotation matrix
         - $R_{ref}$: Reference rotation matrix
           )doc")
      .def("set_lower_bound",
           &FrameRotationBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FrameRotationBound::set_upper_bound,
           nb::arg("upper_bound"));

    nb::class_<FrameLinearVelocityBound, StageConstraints>(
      m,
      "FrameLinearVelocityBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Vec3 &,
                    uint,
                    uint,
                    bool>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("terminal") = false,
           R"doc(
        Predefined 3D velocity bound for a given frame
        Creates a linear bound constraint on the velocity of a frame of the given
        model: $\boldsymbol{b}_l <= \boldsymbol{v}_f <= \boldsymbol{b}_u$
        where:
         - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
         - $\boldsymbol{v}_f$: Frame velocity vector (in $\mathcal{R^3}$)          
           )doc")
      .def("set_lower_bound",
           &FrameLinearVelocityBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FrameLinearVelocityBound::set_upper_bound,
           nb::arg("upper_bound"));
}
}
}
