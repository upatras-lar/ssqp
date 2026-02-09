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
    NB_TRAMPOLINE(StageConstraints, 10);

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
        return (Scalar() << nb::cast<double>(
                  nb_trampoline.base().attr(nb_ticket.key)(x, u)))
          .finished();
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

    nb::class_<FramePositionBound, StageConstraints>(m, "FramePositionBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const VecX &,
                    const VecX &,
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

    nb::class_<FrameVelocityBound, StageConstraints>(m, "FrameVelocityBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const VecX &,
                    const VecX &,
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
           &FrameVelocityBound::set_lower_bound,
           nb::arg("lower_bound"))
      .def("set_upper_bound",
           &FrameVelocityBound::set_upper_bound,
           nb::arg("upper_bound"));
}
}
}
