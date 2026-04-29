#ifndef SSQP_MODEL_STAGEWISE_STAGE_DYNAMICS_HPP
#define SSQP_MODEL_STAGEWISE_STAGE_DYNAMICS_HPP

#include <vector>
#include <unordered_map>

#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
/* General abstract dynamics function for an optimization stage $k$: $\boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
 * Defines the required functions for stage dynamics, while providing 
 * default implementations for some
 */
class StageDynamics : public Function
{
  public:
    StageDynamics();
    StageDynamics(const std::string &name);

    // Size of state vector $\boldsymbol{x}$
    virtual uint state_size() const = 0;

    // Size of control vector $\boldsymbol{u}$
    virtual uint control_size() const = 0;

    // Size of the difference vector: $\boldsymbol{x}_1 \ominus \boldsymbol{x}_2$
    // Where $\ominus$: the defined right_minus operator
    virtual uint state_diff_size() const;

    // Initial state vector (size $n_x$) (default $\boldsymbol{0}$)
    virtual VecX init_state() const;

    // Set the initial state vector
    virtual void set_init_state(const VecX &init_state);

    // Initial control vector (size $n_u$) (default $\boldsymbol{0}$)
    virtual VecX init_control() const;

    // Set the initial control vector
    virtual void set_init_control(const VecX &init_control);

    // Operator definitions for states
    // Defines operators between states and state differences
    // If state is a euclidian vector, these operators reduce to simple
    // addition/subtraction
    // If state is a lie group element, these operators perform conversions
    // between elements of the group (manifold) and tangent space elements
    // The state can also be composite, i.e made up of one or more manifolds and
    // one or more euclidian vectors, in which case operators act segment-wise

    // Defines right plus operator between state vector and tangent space vector
    // Inputs: State vector $\boldsymbol{x}$ (size $n_x$), state difference vector $\boldsymbol{dx}$ (size $n_{dx}$)
    // Output: State $\boldsymbol{y}$ (size $n_x$)
    // Default behavior: $\boldsymbol{x} + \boldsymbol{dx}$
    virtual VecX state_right_plus(const VecX &x, const VecX &dx) const;

    // Defines right minus operator between states
    // Inputs: Composite states $\boldsymbol{x}$, $\boldsymbol{y}$ (vectors/lie group elements, size $n_x$)
    // Output: State difference vector $\boldsymbol{dx}$ (size $n_{dx}$)
    // Default behavior: $\boldsymbol{x} - \boldsymbol{y}$
    virtual VecX state_right_minus(const VecX &x, const VecX &y) const;

    // Get dynamics Jacobians with respect to $\boldsymbol{x}$ and $\boldsymbol{u}$, evaluated at
    // given $(\boldsymbol{x}, \boldsymbol{u})$: $\nabla_x \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$ and $\nabla_u \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  dX: gradient with respect to $\boldsymbol{x}$, $n_{dx} \times n_{dx}$
    //  dU: gradient with respect to $\boldsymbol{u}$, $n_{dx} \times n_u$
    virtual void gradient(const VecX &x, const VecX &u, MatX &dX,
                          MatX &dU) const = 0;

    // Get product $\nabla_{x} \lambda_k^\top \nabla_{x} \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Pxx: Product, $n_{dx} \times n_{dx}$
    virtual void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                             MatX &Pxx) const = 0;

    // Get product $\nabla_{x} \lambda_k^\top \nabla_{u} \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Pxu: Product, $n_{dx} \times n_u$
    virtual void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                             MatX &Pxu) const = 0;

    // Get product $\nabla_{u} \lambda_k^\top \nabla_{u} \boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Puu: Product, $n_u \times n_u$
    virtual void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                             MatX &Puu) const = 0;

  protected:
    static uint TOTAL;

    VecX _init_state, _init_control;
};
}
}
}

#endif