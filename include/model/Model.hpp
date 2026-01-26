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

// TODO: optimize calls to pinocchio algorithms (avoid recomputations)
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
    /* Model Structure struct
     * ---
     * Defines the structure of the SSQP model
     * 
     * A Structure object can be passed to the Model constructor to initialize
     * a model with a structure only, without necessarily finalizing the model
     * 
     * Parameters:
     *  - `urdf_path`: Path to URDF describing robot (may be left empty if
     *     custom dynamics is to be used)
     *  - `free_flyer`: Whether the model has a floating base (only relevant
     *     if initialized through URDF)
     *  - `terrain`: The ground terrain (default flat at z=0)
     *  - `dynamics_mode`: The dynamics mode to use (default Forward Dynamics)
     *  - `integration`: The integration scheme to use (default Semi-Implicit
     *     Euler)
     *  - `ignore_innate_constraints`: Whether to ignore innate constraints from
     *     the URDF (i.e joint position, velocity and torque limits)
     *  - `actuated_joints`: Indices of the actuated joints (default empty ->
     *     all joints actuated)
     */
    struct Structure
    {
        std::string                       urdf_path;
        bool                              free_flyer    = false;
        std::shared_ptr<terrain::Terrain> terrain       = nullptr;
        DynamicsMode                      dynamics_mode = Forward;
        IntegrationScheme                 integration   = SemiImplicitEuler;
        bool                              ignore_innate_constraints = false;
        std::vector<uint>                 actuated_joints;

        static Structure Default()
        {
            return {};
        }
    };

    /* Model Config struct
     * ---
     * Defines the configuration of the SSQP model
     * 
     * A Config object needs to be passed when finalizing the model
     * This can either be done by directly finalizing via the constructor, or
     * by using the finalize method later on
     * 
     * Parameters:
     *  - `costs`: (size $H$) Vector of stage costs (added to existing costs)
     *  - `stage_cost`: Stage cost for all non-terminal stages (added to
     *     existing costs)
     *  - `terminal_cost`: Stage cost for terminal stage
     *  - `constraints`: (empty / size $H + 1$) Vector of stage constraints
     *  - `dynamics`: (empty -> URDF dynamics / size 1 -> same dynamics for
     *     every stage / size $H + 1$) Vector of stage dynamics
     *  - `init_states`: (empty / size $1$ / size $H + 1$) Vector of initial states
     *     (empty -> $\boldsymbol{0}$ initial states / size $1$ -> same initial state for 
     *     every stage / size $H + 1$ -> given sizes per stage)
     *  - `init_controls`: (empty / size $1$ / size $H$) Vector of initial controls
     *     (empty -> $\boldsymbol{0}$ initial controls / size $1$ -> same initial controls for 
     *     every stage / size $H$ -> given controls per stage)
     */
    struct Config
    {
        std::vector<std::shared_ptr<stagewise::StageCost>>        costs;
        std::shared_ptr<stagewise::StageCost>                     stage_cost;
        std::shared_ptr<stagewise::StageCost>                     terminal_cost;
        std::vector<std::shared_ptr<stagewise::StageConstraints>> constraints;
        std::vector<std::shared_ptr<stagewise::StageDynamics>>    dynamics;

        std::vector<VecX> init_states;
        std::vector<VecX> init_controls;

        static Config Default()
        {
            return {};
        }
    };

    /* Model Structure Constructor
     * ---
     * Initializes a model with the given structure
     * 
     * Input:
     *  - `horizon`: The time horizon to optimize over
     *  - `dt`: Time step per solver stage
     *  - `structure`: The structure of the problem
     * 
     * #### Note:
     * This constructor does not finalize the model. The `finalize` method 
     * must be explicitly called before passing the model to a solver. If an
     * unfinalized model is passed to a solver, the solver will attempt to
     * finalize it with the default configuration.
     */
    Model(uint horizon, double dt,
          const Structure &structure = Structure::Default());

    /* Model Config Constructor
     * ---
     * Initializes model with default structure and the given configuration
     * 
     * Input:
     *  - `horizon`: The time horizon to optimize over
     *  - `dt`: Time step per solver stage
     *  - `config`: The system configuration
     * 
     * #### Note:
     * This constructor finalizes the model.
     * When using this constructor to finalize the model, the `finalize` method
     * must not be called afterwards.
     */
    Model(uint horizon, double dt, const Config &config);

    /* Full Model Constructor
     * ---
     * Initializes a model with the given structure and finalizes the model with
     * the given configuration
     * 
     * Input:
     *  - `horizon`: The time horizon to optimize over
     *  - `dt`: Time step per solver stage
     *  - `structure`: The structure of the problem
     *  - `config`: The system configuration
     * 
     * #### Note:
     * This constructor finalizes the model.
     * When using this constructor to finalize the model, the `finalize` method
     * must not be called afterwards.
     */
    Model(uint horizon, double dt, const Structure &structure,
          const Config &config);

    /* `finalize` method
     * ---
     * Finalizes the model with the given configuration
     * 
     * Input:
     *  - `config`: The system configuration
     * 
     * #### Note:
     * This method must only be called once, in the case that the model is not
     * finalized, before passing the model to a solver. If it is not called,
     * the default configuration will be used.
     */
    void finalize(const Config &config = Config::Default());

    // Whether the model is finalized
    bool finalized() const;

    // Whether the model has been loaded from a URDF file
    bool urdf() const;

    const uint       horizon() const;
    const Dimensions dims() const;
    const Dimensions linear_dims() const;

    const CostsContainer       costs() const;
    const DynamicsContainer    dynamics() const;
    const ConstraintsContainer constraints() const;

    const std::shared_ptr<terrain::Terrain> terrain() const;

    // Set initial state for stage k (must be finalized)
    void setInitState(const uint k, const VecX &init_state);

    // Set initial control for stage k (must be finalized)
    void setInitControl(const uint k, const VecX &init_control);

    // Add a new cost for stage k of StageCost type L
    // Returns: the created cost
    template<typename L, typename... Args>
    std::shared_ptr<stagewise::StageCost> addCost(const uint k, Args... args);

    // Add new constraints for stage k of StageConstraint type L
    // Returns: the created constraints
    template<typename C, typename... Args>
    std::shared_ptr<stagewise::StageConstraints> addConstraints(const uint k,
                                                                Args... args);

    // Add a new cost for stage k from existing object
    void addCost(const uint                                   k,
                 const std::shared_ptr<stagewise::StageCost> &cost);

    // Add new constraints for stage k from existing object
    void addConstraints(
      const uint                                          k,
      const std::shared_ptr<stagewise::StageConstraints> &constraints);

    // Add a new external force as an additional decision variable for stage k
    // on the given frame
    // Returns the newly created force
    Force addForce(const uint k, const std::string &frame);

    // Returns the number of external forces added for a given stage
    uint nforces(const uint k) const;

    std::vector<std::string> orderedForceFrames(const uint k) const;

    bool hasFrame(const std::string &frame) const;

    void printFramesAndJoints() const;

    std::vector<std::string> orderedActuatedJoints() const;

    // Below functions are undefined if no URDF was used for initialization
    bool floatingBase() const;
    // Robot state size
    uint nq(const uint k) const;
    // Robot control size
    uint nv(const uint k) const;
    // Number of actual joints in robot (ignores universe, root)
    uint njoints(const uint k) const;
    // Optimization time step
    // If the user passed custom dynamics that integrate this value internally,
    // this might be undefined
    double dt() const;

    // Algorithms defined if URDF was used to initialize model

    const VecX neutralPose() const;

    // Perform forward kinematics with the URDF model dynamics
    const std::pair<VecX, MatX> forwardKinematics(
      const VecX &q, const std::string &frame) const;

    const MatX forwardKinematicsSE3(const VecX        &q,
                                    const std::string &frame) const;

    // Get the velocity for a given frame
    const Vec3 frameVelocity(const VecX &q, const VecX &v,
                             const std::string &frame,
                             bool               local = false) const;

    std::pair<MatX, MatX> frameVelocityDerivatives(const VecX &q, const VecX &v,
                                                   const VecX        &a,
                                                   const std::string &frame,
                                                   bool local = false) const;

    const VecX inverseKinematics(const VecX        &desired_translation,
                                 const MatX        &desired_rotation,
                                 const std::string &joint);

    const VecX inverseDynamics(const VecX &q, const VecX &v,
                               const VecX &a) const;

    const VecX inverseDynamics(
      const VecX &q, const VecX &v, const VecX &a,
      const std::unordered_map<std::string, std::vector<Vec3>> &forces) const;

    const VecX inverseDynamics(const VecX &q, const VecX &v, const VecX &a,
                               const std::vector<Vec6> &joint_forces) const;

    // Get a frame jacobian based on the URDF model dynamics
    const MatX frameJacobian(const VecX &x, const std::string &frame,
                             bool local = false) const;

    // Get frame jacobian (assuming computeJointJacobians has been called)
    const MatX frameJacobian(const std::string &frame,
                             bool               local = false) const;

    void dDifference(const VecX &q0, const VecX &q1, MatX &J,
                     const uint arg_pos) const;

    VecX randomState(const uint stage) const;
    VecX randomControls(const uint stage) const;

  protected:
    // Structure properties
    std::shared_ptr<pinocchio_wrapper::ModelImpl> _impl;
    Dimensions                                    _dims, _lin_dims;
    const double                                  _dt;
    const std::shared_ptr<terrain::Terrain>       _terrain;
    const bool                                    _free_flyer;
    const DynamicsMode                            _dynamics_mode;
    const IntegrationScheme                       _integration;
    const bool                                    _ignore_innate;
    const std::vector<uint>                       _actuated_joints;

    // Configuration properties
    bool                            _finalized = false;
    bool                            _urdf;
    CostsContainer                  _costs;
    DynamicsContainer               _dynamics;
    ConstraintsContainer            _constraints;
    std::vector<std::vector<Force>> _forces;

    // Total joints (including non-actuated)
    uint _total_joints;

    // Helper functions

    /* Calculates joint forces given a configuration and an external force
     * vector
     * Input:
     *  - `q`: The model configuration
     *  - `f`: Stacked vector of external forces (the force part of the control
     *    vector)
     *  - `force_data`: Vector of data about each force in `f`
     *
     * Output: A vector of 6D forces, one for each of the model's joints
     */
    std::vector<Vec6> _calculateJointForces(
      const VecX &q, const VecX &f, const std::vector<Force> &force_data) const;

    /* Calculates joint forces and total force jacobian given a configuration
     * and an external force vector
     *
     * Same as _calculateJointForces, but also calculates the full force
     * jacobian
     */
    std::pair<std::vector<Vec6>, MatX> _calculateJointForcesAndJacobian(
      const VecX &q, const VecX &f, const std::vector<Force> &force_data) const;

    // Functionality functions

    // Adds all joint angle, velocity and effort limits from URDF as constraints
    void _addURDFInnateConstraints();

    // Adds inverse dynamics torque constraints
    void _addIDTorqueConstraints();

    // Builds system dynamics from URDF
    void _buildURDFDynamics(const Config &config);

    VecX _randomVector(const uint size, const double start = -5,
                       const double end = 5) const;

    /* Context for `_makeStageDynamics` for a given stage
     * Data:
     *  - `S`: Selection matrix for stage ($S \tau$)
     *  - `force_data`: Vector containing a Force object for each force
     *  - `init_state`: Initial state for stage
     *  - `init_control`: Initial controls for stage
     */
    struct _StageDynamicsContext
    {
        const MatX         S;
        std::vector<Force> force_data;
        VecX               init_state;
        VecX               init_control;
    };

    /* Makes the dynamics for a given stage with context `ctx` based on the
     * URDF, according to the passed templated options
     * Options:
     *  - `WithForces`: Whether there are additional external forces for this
     *     stage
     *  - `DM`: The dynamics mode that should be used (Forward or Inverse),
     *     determining whether controls represent torques or accelerations
     *  - `IS`: The integration scheme to be used for deriving the next state
     */
    template<bool WithForces, DynamicsMode DM, IntegrationScheme IS>
    std::shared_ptr<stagewise::StageDynamics> _makeStageDynamics(
      const _StageDynamicsContext &ctx) const;

    /* Context for `_makeStageConstraints` for a given stage
     * Data:
     *  - `force_data`: Vector containing a Force object for each force
     */
    struct _StageInnateConstraintsContext
    {
        std::vector<Force> force_data;
    };

    /* Makes the innate constraints for a given stage with context `ctx` based
     * based on the URDF, according to the passed templated options
     * Options:
     *  - `FloatingBase`: Whether the system includes a floating base or not
     *  - `IsTerminal`: Whether the stage is the terminal stage (no controls)
     *  - `DM`: The dynamics mode that should be used
     *  - `WithForces`: Whether there are additional external forces for this
     *     stage
     * 
     * ### Note
     * If the selected dynamics mode is Inverse, this function does not generate
     * the inverse dynamics constraints required by the Inverse dynamics mode. 
     * `_makeStageIDTorqueConstraints` needs to also be called.
     */
    template<bool FloatingBase, bool IsTerminal,
             DynamicsMode DM = DynamicsMode::Forward, bool WithForces = false>
    std::shared_ptr<stagewise::StageConstraints> _makeStageInnateConstraints(
      const _StageInnateConstraintsContext &ctx) const;

    /* Makes the inverse dynamics torque constraints required by the Inverse
     * dynamics mode, based on the URDF effort limits.
     * Options:
     *  - `FloatingBase`: Whether the system includes a floating base or not
     *  - `WithForces`: Whether there are additional external forces for this
     *     stage
     * 
     * ### Note
     * This function will only generate inverse dynamics torque constraints,
     * intended for use with the Inverse dynamics mode. This function must be
     * called regardless of the value of the `ignore_innate_constraints` option,
     * since Inverse mode always requires these constraints.
     */
    template<bool FloatingBase, bool WithForces = false>
    std::shared_ptr<stagewise::StageConstraints> _makeStageIDTorqueConstraints(
      const _StageInnateConstraintsContext &ctx) const;
};
}
}

#include "src/model/Model.tpp"

#endif