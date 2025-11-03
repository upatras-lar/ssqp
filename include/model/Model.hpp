#ifndef SSQP_MODEL_MODEL_HPP
#define SSQP_MODEL_MODEL_HPP

#include "types.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"
#include "model/Container.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageConstraints.hpp"
#include "model/stagewise/Factories.hpp"

namespace ssqp
{
namespace model
{

using CostsContainer       = Container<stagewise::StageCost>;
using DynamicsContainer    = Container<stagewise::StageDynamics>;
using ConstraintsContainer = Container<stagewise::StageConstraints>;

class Model
{
  public:
    /* Non-URDF SSQP Model constructor
     * ---
     * Initializes model from custom dynamics, costs and constraints
     *
     * Input:
     *  - `costs`: (size `HORIZON+1`) vector of stage costs (non-null)
     *  - `dynamics`: (size `HORIZON+1`) vector of stage dynamics (non-null)
     *  - `constraints`: (size `HORIZON+1`) vector of stage constraints
     *     (nullable)
     *
     * Finalizes model: true
     */
    Model(
      const std::vector<std::shared_ptr<stagewise::StageCost>>     &costs,
      const std::vector<std::shared_ptr<stagewise::StageDynamics>> &dynamics,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints = {});

    /* URDF SSQP Model constructor (same cost per stage & terminal cost)
     * ---
     * Initializes model from dynamics as defined in URDF, stage costs `cost`,
     * terminal cost `terminal_cost`, stage constraints `constraints` and
     * initial states `init_states`
     *
     * Input:
     *  - `urdf_path`: Path to urdf file
     *  - `horizon`: Solver horizon
     *  - `dt`: Solver time step
     *  - `cost`: Cost for each non-terminal stage (non-null)
     *  - `terminal_cost`: Cost for terminal stage (non-null)
     *  - `constraints`: (size `HORIZON+1`) vector of stage constraints (default
     *     null -> empty vector)
     *  - `init_states`: (size `HORIZON+1`) vector of initial states for each
     *     stage (default Zero vector -> empty vector)
     *
     * Finalizes model: true
     */
    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::shared_ptr<stagewise::StageCost> &cost,
          const std::shared_ptr<stagewise::StageCost> &terminal_cost,
          const std::vector<std::shared_ptr<stagewise::StageConstraints>>
            &constraints                       = {},
          const std::vector<VecX> &init_states = {});

    /* URDF SSQP Model constructor (different costs per stage & terminal cost)
     * ---
     * Initializes model from dynamics as defined in URDF, stage costs `costs`,
     * terminal cost `terminal_cost`, stage constraints `constraints` and
     * initial states `init_states`
     *
     * Input:
     *  - `urdf_path`: Path to urdf file
     *  - `horizon`: Solver horizon
     *  - `dt`: Solver time step
     *  - `costs`: (size `HORIZON+1`) vector of stage costs (non-null)
     *  - `terminal_cost`: Cost for terminal stage (non-null)
     *  - `constraints`: (size `HORIZON+1`) vector of stage constraints (default
     *     null -> empty vector)
     *  - `init_states`: (size `HORIZON+1`) vector of initial states for each
     *     stage (default Zero vector -> empty vector)
     *
     * Finalizes model: true
     */
    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
          const std::shared_ptr<stagewise::StageCost> &terminal_cost,
          const std::vector<std::shared_ptr<stagewise::StageConstraints>>
            &constraints                       = {},
          const std::vector<VecX> &init_states = {});

    /* URDF SSQP Model constructor (Partial initialization - no costs)
     * ---
     * Initializes model from dynamics as defined in URDF, but without any stage
     * costs
     *
     * Input:
     *  - `urdf_path`: Path to urdf file
     *  - `horizon`: Solver horizon
     *  - `dt`: Solver time step
     *  - `init_states`: (size `HORIZON+1`) vector of initial states for each
     *     stage (default Zero vector -> empty vector)
     *
     * Finalizes model: false
     *
     * ### Note
     * One of the `finalize` methods must be called before model is used for
     * optimization
     */
    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::vector<VecX> &init_states = {});

    /* `finalize` method (same cost per stage & terminal cost)
     * ---
     * Finalizes the model with stage costs `cost`, terminal cost
     * `terminal_cost`, stage constraints `constraints` and initial states
     * `init_states`
     */
    void finalize(
      const std::shared_ptr<stagewise::StageCost> &cost,
      const std::shared_ptr<stagewise::StageCost> &terminal_cost,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints                       = {},
      const std::vector<VecX> &init_states = {});

    /* `finalize` method (different costs per stage & terminal cost)
     * ---
     * Finalizes the model with stage costs `costs`, terminal cost
     * `terminal_cost`, stage constraints `constraints` and initial states
     * `init_states`
     */
    void finalize(
      const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
      const std::shared_ptr<stagewise::StageCost>              &terminal_cost,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints                       = {},
      const std::vector<VecX> &init_states = {});

    /* `finalize` method (no costs)
     * ---
     * Finalizes the model with stage constraints `constraints` and initial
     * states `init_states`
     * Assumes all required stage costs have already been set
     *
     * ### Note
     * If there are unset stage costs, finalization will fail
     */
    void finalize(
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints                       = {},
      const std::vector<VecX> &init_states = {});

    const uint       horizon() const;
    const Dimensions dims() const;
    const Dimensions linear_dims() const;

    const CostsContainer       costs() const;
    const DynamicsContainer    dynamics() const;
    const ConstraintsContainer constraints() const;

    std::shared_ptr<stagewise::StageCost> &cost(const uint k);

    // Perform forward kinematics with the URDF model dynamics
    const std::pair<VecX, MatX> forwardKinematics(
      const VecX &x, const std::string &frame) const;

    const VecX inverseKinematics(const VecX        &desired_translation,
                                 const MatX        &desired_rotation,
                                 const std::string &joint);

    // Get a frame jacobian based on the URDF model dynamics
    const MatX frameJacobian(const VecX &x, const std::string &frame) const;

  protected:
    std::shared_ptr<pinocchio_wrapper::ModelImpl> _impl;

    Dimensions           _dims, _lin_dims;
    CostsContainer       _costs;
    DynamicsContainer    _dynamics;
    ConstraintsContainer _constraints;

    bool _finalized = false;

    void _setup();
};
}
}

#endif