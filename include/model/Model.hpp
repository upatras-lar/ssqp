#ifndef SSQP_MODEL_MODEL_HPP
#define SSQP_MODEL_MODEL_HPP

#include "types.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"
#include "model/Container.hpp"
#include "model/terrain/Terrain.hpp"
#include "model/terrain/predefined/FlatTerrain.hpp"
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

// TODO: check dynamics size (H or H+1)
// TODO: optimize calls to pinocchio algorithms (avoid recomputations)
// TODO: allow finalize to not be called by the user whenever possible
/* Defines an SSQP model
 * An SSQP model object is a prerequisite to performing optimization using SSQP
 * The model contains the following:
 *  - An optimization horizon, $H$
 *  - A cost function for each stage of the horizon, $\ell_k(\boldsymbol{x}, \boldsymbol{u}) \in \mathcal{R}$, $0 \leq k \leq H$
 *  - Dynamics functions for each stage of the horizon, $\boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u}) \in \mathcal{R}^n$, $0 \leq k \leq H$
 *  - Constraint functions for each stage of the horizon, $\boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u}) \in \mathcal{R}^c$, $0 \leq k \leq H$
 * 
 * Of the above, all stage cost and dynamics functions must be defined, but
 * constraints may be left empty.
 * 
 * Note that models optimize for $H$ total steps while considering the initial
 * state to be constant, whereas the initial controls can be altered. This means
 * that there are $H$ total controls ($\boldsymbol{u}_0$ to $\boldsymbol{u}_{H-1}$) and $H + 1$ total states ($\boldsymbol{x}_0$ to 
 * $\boldsymbol{x}_H$), with $\boldsymbol{x}_0$ being constant. At the terminal stage (stage $H$), there are no
 * controls and, as such, stagewise functions must be appropriately defined to
 * ignore them. Predefined functions all have constructors to account for this.
 * 
 * Models may be defined either by passing custom function implementations, or
 * by using URDF files. Some constructors allow for defining a model using a
 * URDF file, but without passing the required cost functions. This is done to
 * allow access to robot-related algorithms such as forward dynamics, which may
 * be used to define model costs. Note that, if such constructors are used, the
 * `finalize` method must be called before attempting to use the model for
 * optimization. The user is responsible for either manually assigning all
 * required costs, or passing them to the `finalize` method.
 */
class Model
{
  public:
    /* Non-URDF SSQP Model constructor
     * ---
     * Initializes model from custom dynamics, costs and constraints
     *
     * Input:
     *  - `costs`: (size $H + 1$) vector of stage costs (non-null)
     *  - `dynamics`: (size $H + 1$) vector of stage dynamics (non-null)
     *  - `constraints`: (size $H + 1$) vector of stage constraints
     *     (nullable)
     *
     * Finalizes model: true
     */
    Model(
      const std::vector<std::shared_ptr<stagewise::StageCost>>     &costs,
      const std::vector<std::shared_ptr<stagewise::StageDynamics>> &dynamics,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints = {},
      const std::shared_ptr<terrain::Terrain> &terrain =
        terrain::make_terrain<terrain::predefined::FlatTerrain>());

    /* URDF SSQP Model constructor (same cost per stage & terminal cost)
     * ---
     * Initializes model from dynamics as defined in URDF, stage costs `cost`,
     * terminal cost `terminal_cost`, stage constraints `constraints`, initial 
     * states `init_states` and initial controls `init_controls`
     *
     * Input:
     *  - `urdf_path`: Path to urdf file
     *  - `horizon`: Solver horizon
     *  - `dt`: Solver time step
     *  - `cost`: Cost for each non-terminal stage (non-null)
     *  - `terminal_cost`: Cost for terminal stage (non-null)
     *  - `constraints`: (size $H + 1$) vector of stage constraints (default
     *     null -> empty vector)
     *  - `init_states`: (size $H + 1$) vector of initial states for each
     *     stage (default $\boldsymbol{0}$ vector -> empty vector)
     *  - `init_controls`: (size $H$) vector of initial controls for each
     *     stage (default $\boldsymbol{0}$ vector -> empty vector)
     *  - `free_flyer`: Whether the model has a floating base (default `false`)
     *  - `terrain`: The ground terrain (default flat at z=0)
     *
     * Finalizes model: true
     */
    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::shared_ptr<stagewise::StageCost> &cost,
          const std::shared_ptr<stagewise::StageCost> &terminal_cost,
          const std::vector<std::shared_ptr<stagewise::StageConstraints>>
            &constraints                         = {},
          const std::vector<VecX> &init_states   = {},
          const std::vector<VecX> &init_controls = {},
          const bool free_flyer                  = false,
          const std::shared_ptr<terrain::Terrain> &terrain =
            terrain::make_terrain<terrain::predefined::FlatTerrain>());

    /* URDF SSQP Model constructor (different costs per stage & terminal cost)
     * ---
     * Initializes model from dynamics as defined in URDF, stage costs `costs`,
     * terminal cost `terminal_cost`, stage constraints `constraints`, initial
     * states `init_states` and initial controls `init_controls`
     *
     * Input:
     *  - `urdf_path`: Path to urdf file
     *  - `horizon`: Solver horizon
     *  - `dt`: Solver time step
     *  - `costs`: (size $H + 1$) vector of stage costs (non-null)
     *  - `terminal_cost`: Cost for terminal stage (non-null)
     *  - `constraints`: (size $H + 1$) vector of stage constraints (default
     *     null -> empty vector)
     *  - `init_states`: (size $H + 1$) vector of initial states for each
     *     stage (default $\boldsymbol{0}$ vector -> empty vector)
     *  - `init_controls`: (size $H$) vector of initial controls for each
     *     stage (default $\boldsymbol{0}$ vector -> empty vector)
     *  - `free_flyer`: Whether the model has a floating base (default `false`)
     *  - `terrain`: The ground terrain (default flat at z=0)
     *
     * Finalizes model: true
     */
    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
          const std::shared_ptr<stagewise::StageCost> &terminal_cost,
          const std::vector<std::shared_ptr<stagewise::StageConstraints>>
            &constraints                         = {},
          const std::vector<VecX> &init_states   = {},
          const std::vector<VecX> &init_controls = {},
          const bool free_flyer                  = false,
          const std::shared_ptr<terrain::Terrain> &terrain =
            terrain::make_terrain<terrain::predefined::FlatTerrain>());

    /* URDF SSQP Model constructor (Partial initialization - no costs)
     * ---
     * Initializes model from dynamics as defined in URDF, but without any stage
     * costs
     *
     * Input:
     *  - `urdf_path`: Path to urdf file
     *  - `horizon`: Solver horizon
     *  - `dt`: Solver time step
     *  - `free_flyer`: Whether the model has a floating base (default `false`)
     *  - `terrain`: The ground terrain (default flat at z=0)
     *
     * Finalizes model: false
     *
     * ### Note
     * One of the `finalize` methods must be called before model is used for
     * optimization
     */
    Model(const std::string &urdf_path, const uint horizon, const double dt,
          const bool free_flyer = false,
          const std::shared_ptr<terrain::Terrain> &terrain =
            terrain::make_terrain<terrain::predefined::FlatTerrain>());

    /* `finalize` method (same cost per stage & terminal cost)
     * ---
     * Finalizes the model with stage costs `cost`, terminal cost
     * `terminal_cost`, stage constraints `constraints`, initial states
     * `init_states` and initial controls `init_controls`
     */
    void finalize(
      const std::shared_ptr<stagewise::StageCost> &cost,
      const std::shared_ptr<stagewise::StageCost> &terminal_cost,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints                         = {},
      const std::vector<VecX> &init_states   = {},
      const std::vector<VecX> &init_controls = {});

    /* `finalize` method (different costs per stage & terminal cost)
     * ---
     * Finalizes the model with stage costs `costs`, terminal cost
     * `terminal_cost`, stage constraints `constraints`, initial states
     * `init_states` and initial controls `init_controls`
     */
    void finalize(
      const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
      const std::shared_ptr<stagewise::StageCost>              &terminal_cost,
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints                         = {},
      const std::vector<VecX> &init_states   = {},
      const std::vector<VecX> &init_controls = {});

    /* `finalize` method (no costs)
     * ---
     * Finalizes the model with stage constraints `constraints`, initial
     * states `init_states` and initial controls `init_controls`
     * Assumes all required stage costs have already been set
     *
     * ### Note
     * If there are unset stage costs, finalization will fail
     */
    void finalize(
      const std::vector<std::shared_ptr<stagewise::StageConstraints>>
        &constraints                         = {},
      const std::vector<VecX> &init_states   = {},
      const std::vector<VecX> &init_controls = {});

    const uint       horizon() const;
    const Dimensions dims() const;
    const Dimensions linear_dims() const;

    const CostsContainer       costs() const;
    const DynamicsContainer    dynamics() const;
    const ConstraintsContainer constraints() const;

    const std::shared_ptr<terrain::Terrain> terrain() const;

    // Add a new cost for stage k of StageCost type L
    template<typename L, typename... Args>
    void addCost(const uint k, Args... args);

    // Add a new constraint for stage k of StageConstraint type L
    template<typename C, typename... Args>
    void addConstraints(const uint k, Args... args);

    // Add a new external force as an additional decision variable for stage k
    // on the given frame
    // Returns the newly created force
    Force addForce(const uint k, const std::string &frame);

    // Returns the number of external forces added for a given stage
    uint nforces(const uint k) const;

    void printFramesAndJoints();

    std::vector<std::string> orderedActuatedJoints() const;

    // Below functions are undefined if no URDF was used for initialization
    bool floating_base() const;
    // Robot state size
    uint nq() const;
    // Robot control size
    uint nv() const;
    // Number of actual joints in robot (ignores universe, root)
    uint njoints() const;
    // Optimization time step
    // If the user passed custom dynamics that integrate this value internally,
    // this might be undefined
    double dt() const;

    // Perform forward kinematics with the URDF model dynamics
    const std::pair<VecX, MatX> forwardKinematics(
      const VecX &q, const std::string &frame) const;

    const MatX forwardKinematicsSE3(const VecX        &q,
                                    const std::string &frame) const;

    // Get the velocity for a given frame
    const Vec3 frameVelocity(const VecX &q, const VecX &v,
                             const std::string &frame,
                             bool               local = false) const;

    std::pair<MatX, MatX> frameVelocityDerivatives(const VecX &x, const VecX &u,
                                                   const std::string &frame,
                                                   bool local = false) const;

    const VecX inverseKinematics(const VecX        &desired_translation,
                                 const MatX        &desired_rotation,
                                 const std::string &joint);

    const VecX inverseDynamics(
      const VecX &q, const VecX &v, const VecX &a,
      const std::unordered_map<std::string, std::vector<Vec3>> &forces) const;

    // Get a frame jacobian based on the URDF model dynamics
    const MatX frameJacobian(const VecX &x, const std::string &frame,
                             bool local = false) const;

    // Get frame jacobian (assuming computeJointJacobians has been called)
    const MatX frameJacobian(const std::string &frame,
                             bool               local = false) const;

  protected:
    std::shared_ptr<pinocchio_wrapper::ModelImpl> _impl;
    uint                                          _nq, _nv, _njoints;
    const double                                  _dt;
    const bool                                    _free_flyer;

    // Selection matrix (for controls)
    MatX _S;

    Dimensions                        _dims, _lin_dims;
    CostsContainer                    _costs;
    DynamicsContainer                 _dynamics;
    ConstraintsContainer              _constraints;
    std::shared_ptr<terrain::Terrain> _terrain;

    std::vector<std::vector<Force>> _forces;

    bool _finalized = false;

    void _constructDynamics(const std::vector<VecX> &init_states   = {},
                            const std::vector<VecX> &init_controls = {});
    void _setup();
};
}
}

#include "src/model/Model.tpp"

#endif