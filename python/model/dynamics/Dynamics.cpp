#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
#include "model/stagewise/StageDynamics.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model;
using namespace model::stagewise;

struct __attribute__((visibility("default"))) PyStageDynamics : StageDynamics
{
    NB_TRAMPOLINE(StageDynamics, 13);

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
              "`StageDynamics::output_size`: expected an integer");
        }
    }

    uint state_size() const override
    {
        try
        {
            NB_OVERRIDE_PURE(state_size);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageDynamics::state_size`: expected an integer");
        }
    }

    uint control_size() const override
    {
        try
        {
            NB_OVERRIDE_PURE(control_size);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageDynamics::control_size`: expected an integer");
        }
    }

    uint state_diff_size() const override
    {
        try
        {
            NB_OVERRIDE(state_diff_size);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageDynamics::state_diff_size`: expected an integer");
        }
    }

    VecX init_state() const override
    {
        try
        {
            NB_OVERRIDE(init_state);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageDynamics::init_state`: expected a vector (numpy.ndarray)");
        }
    }

    VecX init_control() const override
    {
        try
        {
            NB_OVERRIDE(init_control);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageDynamics::init_control`: expected "
                                     "a vector (numpy.ndarray)");
        }
    }

    VecX state_right_plus(const VecX &x, const VecX &dx) const override
    {
        try
        {
            NB_OVERRIDE(state_right_plus, x, dx);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageDynamics::state_right_plus`: "
                                     "expected a vector (numpy.ndarray)");
        }
    }

    VecX state_right_minus(const VecX &x, const VecX &y) const override
    {
        try
        {
            NB_OVERRIDE(state_right_minus, x, y);
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageDynamics::state_right_minus`: "
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
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageDynamics::evaluate`: expected a vector (numpy.ndarray)");
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
              "`StageDynamics::gradient`: expected a pair of matrices "
              "(numpy.ndarray, numpy.ndarray)");
        }
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxx) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessprod_xx",
                                               true);
            Pxx = nb::cast<MatX>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u, lambda));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageDynamics::hessprod_xx`: expected a "
                                     "matrix (numpy.ndarray)");
        }
    }

    void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxu) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessprod_xu",
                                               true);
            Pxu = nb::cast<MatX>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u, lambda));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageDynamics::hessprod_xu`: expected a "
                                     "matrix (numpy.ndarray)");
        }
    }

    void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Puu) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessprod_uu",
                                               true);
            Puu = nb::cast<MatX>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u, lambda));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageDynamics::hessprod_uu`: expected a "
                                     "matrix (numpy.ndarray)");
        }
    }
};

void bind_Dynamics(nb::module_ &m)
{
    nb::class_<StageDynamics, PyStageDynamics>(m, "StageDynamics", R"doc(
        General abstract dynamics function for an optimization stage $k$: $\boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
        Defines the required functions for stage dynamics, while providing 
        default implementations for some
        )doc")
      .def(nb::init())
      .def(nb::init<const std::string &>(), nb::arg("name") = "")
      .def("names",
           &StageDynamics::names,
           "Get all dynamics' names for StageDynamics.")
      .def("output_size",
           &StageDynamics::output_size,
           "Get the dynamics function output size.")
      .def("state_size",
           &StageDynamics::state_size,
           R"doc(Size of state vector $\boldsymbol{x}$.)doc")
      .def("control_size",
           &StageDynamics::control_size,
           R"doc(Size of control vector $\boldsymbol{u}$.)doc")
      .def("state_diff_size", &StageDynamics::state_diff_size, R"doc(
        Size of the difference vector: $\boldsymbol{x}_1 \ominus \boldsymbol{x}_2$
        Where $\ominus$: the defined right_minus operator
        )doc")
      .def("init_state",
           &StageDynamics::init_state,
           R"doc(Initial state vector (size $n_x$) (default $\boldsymbol{0}$).)doc")
      .def("init_control",
           &StageDynamics::init_control,
           R"doc(Initial control vector (size $n_u$) (default $\boldsymbol{0}$).)doc")
      .def("state_right_plus",
           &StageDynamics::state_right_plus,
           nb::arg("x"),
           nb::arg("dx"),
           R"doc(
        Defines right plus operator between state vector and tangent space vector
        Inputs: State vector $\boldsymbol{x}$ (size $n_x$), state difference vector $\boldsymbol{dx}$ (size $n_{dx}$)
        Output: State $\boldsymbol{y}$ (size $n_x$)
        Default behavior: $\boldsymbol{x} + \boldsymbol{dx}$
           )doc")
      .def("state_right_minus",
           &StageDynamics::state_right_minus,
           nb::arg("x"),
           nb::arg("y"),
           R"doc(
        Defines right minus operator between states
        Inputs: Composite states $\boldsymbol{x}$, $\boldsymbol{y}$ (vectors/lie group elements, size $n_x$)
        Output: State difference vector $\boldsymbol{dx}$ (size $n_{dx}$)
        Default behavior: $\boldsymbol{x} - \boldsymbol{y}$
           )doc")
      .def("evaluate",
           &StageDynamics::evaluate,
           nb::arg("x"),
           nb::arg("u"),
           R"doc(
        Evaluate function $f(\boldsymbol{x}, \boldsymbol{u})$
        Returns: output vector, $n \times 1$
            )doc")
      .def(
        "gradient",
        [](const StageDynamics &self, const VecX &x, const VecX &u)
        {
            MatX dX, dU;
            self.gradient(x, u, dX, dU);
            return nb::make_tuple(dX, dU);
        },
        nb::arg("x"),
        nb::arg("u"),
        R"doc(
        Get dynamics Jacobians with respect to $\boldsymbol{x}$ and $\boldsymbol{u}$, evaluated at
        given $(\boldsymbol{x}, \boldsymbol{u})$: $\nabla_x \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$ and $\nabla_u \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         dX: gradient with respect to $\boldsymbol{x}$, $n_{dx} \times n_{dx}$
         dU: gradient with respect to $\boldsymbol{u}$, $n_{dx} \times n_u$
        )doc")
      .def(
        "hessprod_xx",
        [](const StageDynamics &self,
           const VecX &x,
           const VecX &u,
           const VecX &lambda)
        {
            MatX Pxx;
            self.hessprod_xx(x, u, lambda, Pxx);
            return Pxx;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("lam"),
        R"doc(
        Get product $\nabla_{x} \lambda_k^\top \nabla_{x} \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Pxx: Product, $n_{dx} \times n_{dx}$        
        )doc")
      .def(
        "hessprod_xu",
        [](const StageDynamics &self,
           const VecX &x,
           const VecX &u,
           const VecX &lambda)
        {
            MatX Pxu;
            self.hessprod_xu(x, u, lambda, Pxu);
            return Pxu;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("lam"),
        R"doc(
        Get product $\nabla_{x} \lambda_k^\top \nabla_{u} \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Pxu: Product, $n_{dx} \times n_u$        
        )doc")
      .def(
        "hessprod_uu",
        [](const StageDynamics &self,
           const VecX &x,
           const VecX &u,
           const VecX &lambda)
        {
            MatX Puu;
            self.hessprod_uu(x, u, lambda, Puu);
            return Puu;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("lam"),
        R"doc(
        Get product $\nabla_{u} \lambda_k^\top \nabla_{u} \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Puu: Product, $n_u \times n_u$
        )doc")
      .def("parameter_size",
           &StageDynamics::parameter_size,
           "Size of parameter vector $\theta$ (optional method)")
      .def("parameters", &StageDynamics::parameters, R"doc(
        Get parameter vector $\theta$

        Note: Implementing this method is optional; this method allows for online
        adaptation of any model parameters, $\theta$
            )doc")
      .def("parameter_gradient",
           &StageDynamics::parameter_gradient,
           nb::arg("x"),
           nb::arg("u"),
           R"doc(
        Get dynamics Jacobian with respect to any model parameters, evaluated at
        given $(\boldsymbol{x}, \boldsymbol{u})$: $\nabla_{\theta} \boldsymbol{f}_{\theta, k}(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         gradient with respect to $\theta$, $n_{dx} \times n_{\theta}$

        Note: Implementing this method is optional; this method allows for online
        adaptation of any model parameters, $\theta$    
            )doc")
      .def("update_parameters",
           &StageDynamics::update_parameters,
           nb::arg("theta"),
           R"doc(
        Update parameter vector $\theta$
            
        Note: Implementing this method is optional; this method allows for online
        adaptation of any model parameters, $\theta$     
            )doc");
}
}
}
