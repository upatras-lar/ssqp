#ifndef SSQP_MODEL_MODEL_HPP
#define SSQP_MODEL_MODEL_HPP

#include <unordered_set>
#include <variant>
#include <type_traits>
#include <utility>
#include <stdexcept>

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

auto to_compile_time(bool b);
auto to_compile_time(SolverMode sm);
auto to_compile_time(IntegrationScheme is);

// TODO: optimize calls to pinocchio algorithms (avoid recomputations)
/** Defines an SSQP model
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
 * by using XML description files. Some constructors allow for defining a model
 * using an XML file, but without passing the required cost functions. This is 
 * done to allow access to robot-related algorithms such as forward dynamics,
 * which may be used to define model costs. Note that, if such constructors are
 * used, the `finalize` method must be called before attempting to use the model
 * for optimization. The user is responsible for either manually assigning all
 * required costs, or passing them to the `finalize` method.
 */
class Model
{
  public:
    /** Model Structure struct
     * ---
     * Defines the structure of the SSQP model
     * 
     * A Structure object can be passed to the Model constructor to initialize
     * a model with a structure only, without necessarily finalizing the model
     * 
     * Parameters:
     *  - `urdf_path`: Path to URDF describing robot (may be left empty if other
     *     description is used)
     *  - `mjcf_path`: Path to MJCF describing robot (may be left empty if other
     *     description is used)
     *  - `free_flyer`: Whether the model has a floating base (only relevant
     *     if initialized through XML)
     *  - `from_preset`: A URDF preset to build the model from (default 
     *     `Preset::None`); mutually exclusive with `urdf_path`, `mjcf_path` 
     *     and `free_flyer`
     *  - `terrain`: The ground terrain (default flat at z=0)
     *  - `solver_mode`: The solver mode to use (default `ForwardDynamics`)
     *  - `integration`: The integration scheme to use (default Semi-Implicit
     *     Euler)
     *  - `ignore_innate_constraints`: Whether to ignore innate constraints from
     *     the XML (i.e joint position, velocity and torque limits)
     *  - `actuated_joints`: Indices of the actuated joints (default empty ->
     *     all joints actuated)
     */
    struct Structure
    {
        std::string                       urdf_path;
        std::string                       mjcf_path;
        bool                              free_flyer  = false;
        Preset                            from_preset = Preset::None;
        std::shared_ptr<terrain::Terrain> terrain     = nullptr;
        SolverMode                        solver_mode = ForwardDynamics;
        IntegrationScheme                 integration = SemiImplicitEuler;
        bool                              ignore_innate_constraints = false;
        std::vector<uint>                 actuated_joints;

        static Structure Default()
        {
            return {};
        }
    };

    /** Model Config struct
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
     *  - `dynamics`: (empty -> XML dynamics / size 1 -> same dynamics for
     *     every stage / size $H + 1$) Vector of stage dynamics
     *  - `custom_lower_position_bounds`: Lower position bounds to use in place
     *     of the values read from the XML; default XML values are used if left
     *     empty; has no effect if innate constraints are ignored
     *  - `custom_upper_position_bounds`: Upper position bounds to use in place
     *     of the values read from the XML; default XML values are used if left
     *     empty; has no effect if innate constraints are ignored
     *  - `custom_velocity_bounds`: Velocity bounds to use in place of the 
     *     values read from the XML; default XML values are used if left empty; 
     *     has no effect if innate constraints are ignored
     *  - `custom_torque_bounds`: Torque bounds to use in place of the values
     *     read from the XML; default XML values are used if left empty; has no
     *     effect if innate constraints are ignored
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

        VecX custom_lower_position_bounds, custom_upper_position_bounds;
        VecX custom_velocity_bounds;
        VecX custom_torque_bounds;

        std::vector<VecX> init_states;
        std::vector<VecX> init_controls;

        static Config Default()
        {
            return {};
        }
    };

    /** CustomProperties struct for Model
    * Defines common XML-based properties (like nq & nv) for Models that are not
    * loaded via XML
    */
    struct CustomProperties
    {
        bool                     floating_base = false;
        uint                     nq            = 0;
        uint                     nv            = 0;
        uint                     nu            = 0;
        uint                     njoints       = 0;
        std::vector<std::string> ordered_joints;
        std::vector<std::string> joint_types;
    };

    /** Model Structure Constructor
     * ---
     * Initializes a model with the given structure
     * 
     * Input:
     *  - `horizon`: The time horizon to optimize over
     *  - `dt`: Time step per solver stage
     *  - `structure`: The structure of the problem
     * 
     * @note
     * This constructor does not finalize the model. The `finalize` method 
     * must be explicitly called before passing the model to a solver. If an
     * unfinalized model is passed to a solver, the solver will attempt to
     * finalize it with the default configuration.
     */
    Model(uint horizon, double dt,
          const Structure &structure = Structure::Default());

    /** Model Config Constructor
     * ---
     * Initializes model with default structure and the given configuration
     * 
     * Input:
     *  - `horizon`: The time horizon to optimize over
     *  - `dt`: Time step per solver stage
     *  - `config`: The system configuration
     * 
     * @note
     * This constructor finalizes the model.
     * When using this constructor to finalize the model, the `finalize` method
     * must not be called afterwards.
     */
    Model(uint horizon, double dt, const Config &config);

    /** Full Model Constructor
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
     * @note
     * This constructor finalizes the model.
     * When using this constructor to finalize the model, the `finalize` method
     * must not be called afterwards.
     */
    Model(uint horizon, double dt, const Structure &structure,
          const Config &config);

    // Costs and constraints refer to the model they were built with, so it
    // cannot be copied
    Model(const Model &)            = delete;
    Model &operator=(const Model &) = delete;

    /** finalize method
     * ---
     * Finalizes the model with the given configuration
     * 
     * Input:
     *  - `config`: The system configuration
     * 
     * @note
     * This method must only be called once, in the case that the model is not
     * finalized, before passing the model to a solver. If it is not called,
     * the default configuration will be used.
     */
    void finalize(const Config &config = Config::Default());

    // Whether the model is finalized
    bool finalized() const;

    // Whether the model has been loaded from an XML file
    bool xml() const;

    const uint        horizon() const;
    const Dimensions &dims() const noexcept;
    const Dimensions &linear_dims() const noexcept;

    /** Updates the model's dimensions and linear dimensions
     * This is required to update constraint sizes, when constraints' enabled
     * state changes
     * 
     * TODO: this currently assumes only constraint sizes can change
     */
    void updateDimensions();

    const CostsContainer       &costs() const;
    const DynamicsContainer    &dynamics() const;
    const ConstraintsContainer &constraints() const;

    // Every cost, constraint and dynamics function given to the model, as
    // given, since each stage merges its functions into one
    const std::vector<std::shared_ptr<stagewise::Function>> &functions() const;

    const std::shared_ptr<terrain::Terrain> terrain() const;

    // Set initial state for stage k (must be finalized)
    void setInitState(const uint k, const VecX &init_state);

    // Set initial control for stage k (must be finalized)
    void setInitControl(const uint k, const VecX &init_control);

    // Add a new cost for stage k of StageCost type L
    // Returns: the created cost
    template<typename L, typename... Args>
    std::shared_ptr<stagewise::StageCost> addCost(const uint k, Args &&...args);

    // Add new constraints for stage k of StageConstraint type L
    // Returns: the created constraints
    template<typename C, typename... Args>
    std::shared_ptr<stagewise::StageConstraints> addConstraints(const uint k,
                                                                Args &&...args);

    // Add a new cost for stage k from existing object
    void addCost(const uint                                   k,
                 const std::shared_ptr<stagewise::StageCost> &cost);

    // Add a new running cost of StageCost type L
    // A running cost is a cost added to every non-terminal stage
    template<typename L, typename... Args>
    std::shared_ptr<stagewise::StageCost> addRunningCost(Args &&...args);

    // Add a new running cost from existing object
    // A running cost is a cost added to every non-terminal stage
    void addRunningCost(const std::shared_ptr<stagewise::StageCost> &cost);

    // Add new constraints for stage k from existing object
    void addConstraints(
      const uint                                          k,
      const std::shared_ptr<stagewise::StageConstraints> &constraints);

    // Add a new external linear force (ForceDim::Linear) as an additional
    // decision variable for stage k on the given frame
    // Returns the newly created force
    std::shared_ptr<Force> addForce(const uint k, const std::string &frame);

    // Same as addForce, but the added force is a full spatial wrench
    // (ForceDim::Wrench): a linear force plus a moment/torque, e.g. rotor
    // thrust plus reaction/drag torque
    // Returns the newly created force
    std::shared_ptr<Force> addWrench(const uint k, const std::string &frame);

    // Returns the number of linear forces added for a given stage
    uint nforces(const uint k) const;

    // Returns the number of wrenches added for a given stage
    uint nwrenches(const uint k) const;

    // Vector containing frame names, ordered 1-1 with forces for stage k
    std::vector<std::string> orderedForceFrames(const uint k) const;

    // Vector containing frame names, ordered 1-1 with wrenches for stage k
    std::vector<std::string> orderedWrenchFrames(const uint k) const;

    // Whether the model has a frame with the given name
    bool hasFrame(const std::string &frame) const;

    // Resolves a frame name to its index once; prefer this plus the
    // frame-ID-taking overloads below over repeatedly passing a frame name to
    // hot-path calls, since name resolution is a linear scan
    uint frameId(const std::string &frame) const;

    // Output all frames and joints of the model
    void printFramesAndJoints() const;

    // Ordered frames that are part of type BODY or OP_FRAME
    std::vector<std::string> orderedBodyFrames() const;

    // Types for each joint that is part of the state (including unactuated)
    std::vector<std::string> jointTypes() const;

    // Ordered joints that are part of the state (including unactuated)
    std::vector<std::string> orderedJoints() const;

    // Ordered joints that are part of the state (excluding unactuated)
    std::vector<std::string> orderedActuatedJoints() const;

    // Set the actuated joint indices vector
    void setActuatedJoints(const std::vector<uint> &actuated_joints);

    // Below functions are undefined if no XML was used for initialization and
    // no custom properties exist
    bool floatingBase() const;
    // Robot state size
    uint nq() const;
    // Robot velocity size
    uint nv() const;
    // Robot control size
    uint nu() const;
    // Number of actual joints in robot (ignores universe, root)
    uint njoints() const;
    // Number of total joints in robot
    uint total_joints() const;
    // Optimization time step
    // If the user passed custom dynamics that integrate this value internally,
    // this might be undefined
    double dt() const;
    // The solver mode formulation
    SolverMode solverMode() const;

    // Algorithms defined if XML was used to initialize model

    // Return the neutral pose for the model
    const VecX neutralPose() const;

    // Perform forward kinematics with the XML model dynamics
    const std::pair<Vec3, Mat3> forwardKinematics(
      const VecX &q, const std::string &frame) const;

    const std::pair<Vec3, Mat3> forwardKinematics(const VecX &q,
                                                  const uint  frame_id) const;

    const Mat4 forwardKinematicsSE3(const VecX        &q,
                                    const std::string &frame) const;

    const Mat4 forwardKinematicsSE3(const VecX &q, const uint frame_id) const;

    // Get the velocity for a given frame
    const Vec6 frameVelocity(const VecX &q, const VecX &v,
                             const std::string &frame,
                             bool               local = false) const;

    const Vec6 frameVelocity(const VecX &q, const VecX &v, const uint frame_id,
                             bool local = false) const;

    std::pair<MatX, MatX> frameVelocityDerivatives(const VecX &q, const VecX &v,
                                                   const std::string &frame,
                                                   bool local = false) const;

    std::pair<MatX, MatX> frameVelocityDerivatives(const VecX &q, const VecX &v,
                                                   const uint frame_id,
                                                   bool local = false) const;

    // Output-buffer version of the above; `dVdq`/`dVdv` must already be sized
    // (6 x nv)
    void frameVelocityDerivatives(const VecX &q, const VecX &v,
                                  const std::string &frame, bool local,
                                  Eigen::Ref<MatX> dVdq,
                                  Eigen::Ref<MatX> dVdv) const;

    void frameVelocityDerivatives(const VecX &q, const VecX &v,
                                  const uint frame_id, bool local,
                                  Eigen::Ref<MatX> dVdq,
                                  Eigen::Ref<MatX> dVdv) const;

    const VecX inverseKinematics(const VecX        &desired_translation,
                                 const MatX        &desired_rotation,
                                 const std::string &joint);

    const VecX forwardDynamics(const VecX &q, const VecX &v,
                               const VecX &tau) const;

    const VecX forwardDynamics(
      const VecX &q, const VecX &v, const VecX &tau,
      const std::unordered_map<std::string, std::vector<Vec3>> &forces) const;

    const VecX forwardDynamics(const VecX &q, const VecX &v, const VecX &tau,
                               const std::vector<Vec6> &joint_forces) const;

    const VecX inverseDynamics(const VecX &q, const VecX &v,
                               const VecX &a) const;

    const VecX inverseDynamics(
      const VecX &q, const VecX &v, const VecX &a,
      const std::unordered_map<std::string, std::vector<Vec3>> &forces) const;

    const VecX inverseDynamics(const VecX &q, const VecX &v, const VecX &a,
                               const std::vector<Vec6> &joint_forces) const;

    const void inverseDynamicsDerivatives(
      const Eigen::VectorXd &q, const Eigen::VectorXd &v,
      const Eigen::VectorXd &a, Eigen::MatrixXd &rnea_partial_dq,
      Eigen::MatrixXd &rnea_partial_dv, Eigen::MatrixXd &rnea_partial_da) const;

    const void inverseDynamicsDerivatives(
      const Eigen::VectorXd &q, const Eigen::VectorXd &v,
      const Eigen::VectorXd                       &a,
      const std::vector<Eigen::Vector<double, 6>> &fext,
      Eigen::MatrixXd &rnea_partial_dq, Eigen::MatrixXd &rnea_partial_dv,
      Eigen::MatrixXd &rnea_partial_da) const;

    // Get a frame jacobian based on the XML model dynamics
    const MatX frameJacobian(const VecX &q, const std::string &frame,
                             bool local = false) const;

    const MatX frameJacobian(const VecX &q, const uint frame_id,
                             bool local = false) const;

    // Output-buffer version of the above; `J` must already be sized (6 x nv)
    void frameJacobian(const VecX &q, const std::string &frame, bool local,
                       Eigen::Ref<MatX> J) const;

    void frameJacobian(const VecX &q, const uint frame_id, bool local,
                       Eigen::Ref<MatX> J) const;

    // Get frame jacobian (assuming computeJointJacobians has been called)
    const MatX frameJacobian(const std::string &frame,
                             bool               local = false) const;

    const MatX frameJacobian(const uint frame_id, bool local = false) const;

    // Output-buffer version of the above; `J` must already be sized (6 x nv)
    void frameJacobian(const std::string &frame, bool local,
                       Eigen::Ref<MatX> J) const;

    void frameJacobian(const uint frame_id, bool local,
                       Eigen::Ref<MatX> J) const;

    // Get model center of mass for the given configuration
    const Vec3 centerOfMass(const VecX &q) const;

    // Get model center of mass jacobian for the given configuration
    const MatX jacobianCenterOfMass(const VecX &q) const;

    /** Returns a stacked vector of the current values of the loaded model's
     * dynamic parameters for every link
     * This consist of n vectors of size 10, containing the following:
     * $\boldsymbol{v} = [m, m_{c,x}, m_{c,y}, m_{c_z}, I_{xx}, I_{xy}, I_{yy}, I_{xz}, I_{yz}, I_{zz}]^\top$ for each link
     */
    const VecX getAllDynamicParameters() const;

    /** Sets the dynamic parameters for every link of the loaded model to the
     * provided stacked vector
     * Input:
     *  - `parameters`: A stacked vector of the dynamic parameters for each 
     *     link of the model
     */
    void setAllDynamicParameters(const VecX &parameters);

    /** Computes and returns the joint torque regressor matrix linking the joint 
     * torque to the dynamic parameters of each link for the given 
     * configuration, velocity and acceleration
     * Input:
     *  - `q`: The model configuration
     *  - `v`: The model velocity
     *  - `a`: The model acceleration
     * 
     * Returns: The joint torque regressor matrix
     */
    const MatX getJointTorqueRegressor(const VecX &q, const VecX &v,
                                       const VecX &a) const;

    VecX difference(const VecX &q0, const VecX &q1) const;

    void dDifference(const VecX &q0, const VecX &q1, MatX &J,
                     const uint arg_pos) const;

    // Return the actuated part (nu x nu) of a matrix (nv x nv)
    MatX getMatrixActuatedPart(const MatX &M) const;

    // Output-buffer version of getMatrixActuatedPart; `out` must already be
    // sized (nu x nu) and is written in place, avoiding an allocation
    void getMatrixActuatedPart(const MatX &M, Eigen::Ref<MatX> out) const;

    /** Returns the full acceleration vector for a given stage
     * Input:
     *  - `x`: The model state
     *  - `u`: The model controls
     *  - `k`: The model stage (only required if model is in ReducedInverseDynamics
     *     mode and there are present forces)
     * 
     * Note: For InverseDynamics mode, this function is trivial, since it only
     * returns the non-force part of the control vector
     * Note: XML-based function
     */
    VecX getFullAccelerationVector(const VecX &x, const VecX &u,
                                   const uint k = 0) const;

    /** Returns the actuated-joint torque vector implied by the inverse
     * dynamics computation for the given state and controls, accounting for
     * any forces present at stage `k`
     *
     * For InverseDynamics mode, this is the RNEA torque computed directly
     * from the (full) acceleration control. For ReducedInverseDynamics mode,
     * the full acceleration vector is reconstructed first (see
     * getFullAccelerationVector), and only the actuated joints' torque is
     * returned, since by construction the unactuated joints' torque is
     * always zero
     *
     * Input:
     *  - `x`: The model state
     *  - `u`: The model controls
     *  - `k`: The model stage (only required if there are forces present at
     *     that stage)
     *
     * Note: XML-based function
     * Note: Assumes the model is in InverseDynamics or ReducedInverseDynamics
     *   mode
     */
    VecX getActuatedTorqueVector(const VecX &x, const VecX &u,
                                 const uint k = 0) const;

    /** Derivatives of getActuatedTorqueVector with respect to the state (x)
     * and controls (u)
     *
     * Output:
     *  - `dtau_dx`: derivative with respect to the state (nu x 2*nv)
     *  - `dtau_du`: derivative with respect to the controls (nu x u.size())
     */
    void getActuatedTorqueVectorDerivatives(const VecX &x, const VecX &u,
                                            MatX &dtau_dx, MatX &dtau_du,
                                            const uint k = 0) const;

    VecX randomState(const uint stage) const;
    VecX randomControls(const uint stage) const;

    // Set custom properties for non-XML models
    void setCustomProperties(const CustomProperties &custom_properties);

    // Whether custom properties have been set
    bool hasCustomProperties() const;

  protected:
    // Structure properties
    std::shared_ptr<pinocchio_wrapper::ModelImpl> _impl;
    Dimensions                                    _dims, _lin_dims;
    const double                                  _dt;
    const std::shared_ptr<terrain::Terrain>       _terrain;
    bool                                          _free_flyer;
    const SolverMode                              _solver_mode;
    const IntegrationScheme                       _integration;
    const bool                                    _ignore_innate;

    // Indices of joints that are actuated
    std::vector<uint> _actuated_joints;
    // Complement of `actuated_joints`
    std::vector<uint> _unactuated_joints;
    // Indices of joints that are bounded with constraints
    std::vector<uint> _bounded_joint_indices;
    // Map mapping joint indices to their offset in the configuration vector
    std::unordered_map<uint, uint> _bounded_joint_indices_to_config;
    // Map mapping joint indices to their offset in the tangent vector
    std::unordered_map<uint, uint> _bounded_joint_indices_to_tangent;

    // Configuration properties
    bool                                             _finalized = false;
    bool                                             _xml;
    CostsContainer                                   _costs;
    DynamicsContainer                                _dynamics;
    ConstraintsContainer                             _constraints;
    std::vector<std::shared_ptr<stagewise::Function>> _functions;
    std::vector<std::vector<std::shared_ptr<Force>>> _forces;
    std::vector<std::vector<std::shared_ptr<Force>>> _wrenches;

    // Optional custom properties for non-XML models
    bool             _has_custom_properties = false;
    CustomProperties _custom_properties {};

    // Helper functions

    // Shared implementation of addForce/addWrench: pushes a new Force of the
    // given dimension onto _forces[k] (Linear) or _wrenches[k] (Wrench)
    std::shared_ptr<Force> _addForce(const uint k, const std::string &frame,
                                     const ForceDim dim);

    // Total control-vector width occupied by stage k's forces and wrenches,
    // i.e. `3 * nforces(k) + 6 * nwrenches(k)`
    uint _forceControlWidth(const uint k) const;

    /** Calculates joint forces given a configuration and an external force
     * vector
     * Input:
     *  - `q`: The model configuration
     *  - `f`: Stacked vector of external forces (the force part of the
     *    control vector): linear forces first, then wrenches
     *  - `force_data`: Vector of data about each linear force in `f`
     *  - `wrench_data`: Vector of data about each wrench in `f`
     *
     * Output:
     *  - `joint_forces`: A vector of 6D forces, one for each of the model's
     *     joints; must already be sized (`_dims.total_joints`)
     */
    void _calculateJointForces(
      const VecX &q, const VecX &f,
      const std::vector<std::shared_ptr<Force>> &force_data,
      const std::vector<std::shared_ptr<Force>> &wrench_data,
      std::vector<Vec6>                         &joint_forces) const;

    /** Calculates joint forces and total force jacobian given a configuration
     * and an external force vector
     *
     * Same as _calculateJointForces, but also calculates the full force
     * jacobian
     *
     * Output:
     *  - `joint_forces`: Same as in _calculateJointForces
     *  - `Jf_full`: The full force jacobian; must already be sized
     *     (`3 * force_data.size() + 6 * wrench_data.size()` x `_dims.nv`)
     */
    void _calculateJointForcesAndJacobian(
      const VecX &q, const VecX &f,
      const std::vector<std::shared_ptr<Force>> &force_data,
      const std::vector<std::shared_ptr<Force>> &wrench_data,
      std::vector<Vec6> &joint_forces, MatX &Jf_full) const;

    /** Persistent scratch for the ReducedInverseDynamics mode's acceleration
     * and acceleration-derivatives helpers (`_calculateReducedInverseFullAcceleration`/
     * `_calculateReducedInverseFullAccelerationDerivatives`), sized once per
     * stage and reused across repeated evaluate()/gradient() calls instead of
     * being reallocated every call
     * Bundled into a single struct so the two functions don't need a long
     * individual output-parameter list
     *
     * `M`, `Muu`, `Muj`, `Muu_ldlt`: used/populated by
     * `_calculateReducedInverseFullAcceleration`
     *
     * `dtau_dq`, `dtau_dv`, `dtau_da`, `dtau_daj`, `da_dq`, `da_dv`,
     * `da_daj`, `da_df`: outputs of
     * `_calculateReducedInverseFullAccelerationDerivatives`
     *
     * `Ju`, `dtau_dq_unact`, `dtau_dv_unact`, `da_dq_reordered`,
     * `da_dv_reordered`, `da_daj_reordered`, `da_df_reordered`: internal-only
     * temporaries of `_calculateReducedInverseFullAccelerationDerivatives`
     */
    struct _ReducedInverseDynamicsScratch
    {
        MatX                             M;
        MatX                             Muu, Muj;
        std::optional<Eigen::LDLT<MatX>> Muu_ldlt;

        MatX dtau_dq, dtau_dv, dtau_da, dtau_daj;
        MatX da_dq, da_dv, da_daj, da_df;

        MatX Ju;
        MatX dtau_dq_unact, dtau_dv_unact;
        MatX da_dq_reordered, da_dv_reordered, da_daj_reordered,
          da_df_reordered;
    };

    /** Calculates the full acceleration vector for ReducedInverseDynamics mode
     * Input:
     *  - `q`: The configuration vector
     *  - `v`: The velocity vector
     *  - `ctrl`: The non-force part of the control vector; this is assumed to
     *     be the accelerations for each of the actuated joints
     *  - `f`: The force part of the control vector
     *  - `force_data`: Vector of data about each linear force in `f`
     *  - `wrench_data`: Vector of data about each wrench in `f`
     *  - `actuated_joints`: Vector of indices of joints that are actuated
     *  - `unactuated_joints`: Complement vector of `actuated_joints`
     *
     * Output:
     *  - `scratch`: Populates `scratch.M`, `scratch.Muu`, `scratch.Muj` and
     *     `scratch.Muu_ldlt` (see `_ReducedInverseDynamicsScratch`)
     *  - `joint_forces`: The forces on each joint
     *  - `Jf_full`: The full force jacobian
     *
     * Returns:
     *  - `ddq`: The full acceleration vector
     */
    template<bool FloatingBase, bool WithForces>
    VecX _calculateReducedInverseFullAcceleration(
      const VecX &q, const VecX &v, const VecX &ctrl, const VecX &f,
      const std::vector<std::shared_ptr<Force>> &force_data,
      const std::vector<std::shared_ptr<Force>> &wrench_data,
      const std::vector<uint>                   &actuated_joints,
      const std::vector<uint>                   &unactuated_joints,
      _ReducedInverseDynamicsScratch &scratch, std::vector<Vec6> &joint_forces,
      MatX &Jf_full) const;

    /** Calculates the derivatives of the acceleration vector for ReducedInverseDynamics
     * mode, with respect to q, v, aj (the actual actuated controls) and f
     * Input:
     *  - `q`: The configuration vector
     *  - `v`: The velocity vector
     *  - `ctrl`: The non-force part of the control vector; this is assumed to
     *     be the accelerations for each of the actuated joints
     *  - `f`: The force part of the control vector
     *  - `force_data`: Vector of data about each linear force in `f`
     *  - `wrench_data`: Vector of data about each wrench in `f`
     *  - `actuated_joints`: Vector of indices of joints that are actuated
     *  - `unactuated_joints`: Complement vector of `actuated_joints`
     *
     * Inputs from `_calculateReducedInverseFullAcceleration`'s output:
     *  - `ddq`: The full acceleration vector
     *  - `scratch.Muu`, `scratch.Muj`, `scratch.Muu_ldlt`: See
     *     `_calculateReducedInverseFullAcceleration`'s output
     *  - `joint_forces`: The forces on each joint
     *  - `Jf_full`: The full force jacobian
     *
     * Output:
     *  - `scratch.dtau_dq`: The RNEA derivative with respect to the
     *     configuration (q)
     *  - `scratch.dtau_dv`: The RNEA derivative with respect to the velocity
     *     (v)
     *  - `scratch.dtau_da`: The RNEA derivative with respect to the full
     *     acceleration (a)
     *  - `scratch.dtau_daj`: The RNEA derivative with respect to the actuated
     *     part of the acceleration (aj)
     *  - `scratch.da_dq`: The derivative of the full acceleration vector with
     *     respect to the configuration (q)
     *  - `scratch.da_dv`: The derivative of the full acceleration vector with
     *     respect to the velocity (v)
     *  - `scratch.da_daj`: The derivative of the full acceleration vector
     *     with respect to the actuated part of the acceleration
     *  - `scratch.da_df`: The derivative of the full acceleration vector with
     *     respect to the forces
     */
    template<bool FloatingBase, bool WithForces>
    void _calculateReducedInverseFullAccelerationDerivatives(
      const VecX &q, const VecX &v, const VecX &ctrl, const VecX &f,
      const std::vector<std::shared_ptr<Force>> &force_data,
      const std::vector<std::shared_ptr<Force>> &wrench_data,
      const std::vector<uint>                   &actuated_joints,
      const std::vector<uint> &unactuated_joints, const VecX &ddq,
      _ReducedInverseDynamicsScratch &scratch,
      const std::vector<Vec6> &joint_forces, const MatX &Jf_full) const;

    /** Derivatives of getActuatedTorqueVectorDerivatives for
     * ReducedInverseDynamics mode: composes
     * _calculateReducedInverseFullAcceleration/Derivatives with the RNEA
     * torque derivatives (see getActuatedTorqueVectorDerivatives), then
     * restricts to the actuated joints' rows
     */
    template<bool FloatingBase, bool WithForces>
    void _getActuatedTorqueVectorReducedDerivatives(const VecX &x,
                                                    const VecX &u, const uint k,
                                                    MatX &dtau_dx,
                                                    MatX &dtau_du) const;

    // Functionality functions

    // Adds all joint angle, velocity and effort limits from XML as constraints
    void _buildXMLInnateConstraints(const Config &config);

    // Adds inverse dynamics torque constraints
    void _buildXMLIDTorqueConstraints(const Config &config);

    // Builds system dynamics from XML
    void _buildXMLDynamics(const Config &config);

    VecX _randomVector(const uint size, const double start = -5,
                       const double end = 5) const;

    /** Context for `_makeStageDynamics` for a given stage
     * Data:
     *  - `force_data`: Vector containing a Force object for each linear force
     *  - `wrench_data`: Vector containing a Force object for each wrench
     *  - `init_state`: Initial state for stage
     *  - `init_control`: Initial controls for stage
     *  - `nq`: The configuration size for this stage
     *  - `nv`: The velocity size for this stage
     *  - `nu`: The size of the non-force part of the controls for this stage
     *  - `nf`: The number of linear forces for this stage
     *  - `nw`: The number of wrenches for this stage
     *  - `force_width`: The size of the force part of the controls for this
     *     stage, i.e. `3 * nf + 6 * nw`
     *  - `actuated_joints`: Vector of indices of joints that are actuated
     *  - `unactuated_joints`: Complement vector of `actuated_joints`
     *  - `floating_base`: Whether the model has a floating base for this stage
     */
    struct _StageDynamicsContext
    {
        std::vector<std::shared_ptr<Force>> force_data;
        std::vector<std::shared_ptr<Force>> wrench_data;
        VecX                                init_state;
        VecX                                init_control;
        uint                                nq, nv, nu, nf, nw, force_width;
        std::vector<uint>                   actuated_joints, unactuated_joints;
        std::string                         stage_dynamics_name;

        // Persistent joint-force/force-jacobian scratch, sized once per
        // stage; reused across repeated evaluate()/gradient() calls instead
        // of being reallocated every call. `mutable` since it's written to
        // from the (const) evaluate()/gradient() callbacks
        mutable std::vector<Vec6> joint_forces_scratch;
        mutable MatX              jf_full_scratch;

        // See _ReducedInverseDynamicsScratch; only populated/used in the
        // ReducedInverseDynamics solver mode
        mutable _ReducedInverseDynamicsScratch reduced_id_scratch;
    };

    /** Makes the dynamics for a given stage with context `ctx` based on the
     * XML, according to the passed templated options
     * Options:
     *  - `FloatingBase`: Whether the model has a floating base
     *  - `WithForces`: Whether there are additional external forces for this
     *     stage
     *  - `SM`: The solver mode that should be used
     *  - `IS`: The integration scheme to be used for deriving the next state
     */
    template<bool FloatingBase, bool WithForces, SolverMode SM,
             IntegrationScheme IS>
    std::shared_ptr<stagewise::StageDynamics> _makeStageDynamics(
      const std::shared_ptr<const _StageDynamicsContext> &ctx) const;

    /** Context for `_makeStageConstraints` for a given stage
     * Data:
     *  - `force_data`: Vector containing a Force object for each linear force
     *  - `wrench_data`: Vector containing a Force object for each wrench
     *  - `nq`: The configuration size for this stage
     *  - `nv`: The velocity size for this stage
     *  - `nu`: The size of the non-force part of the controls for this stage
     *  - `nf`: The number of linear forces for this stage
     *  - `nw`: The number of wrenches for this stage
     *  - `force_width`: The size of the force part of the controls for this
     *     stage, i.e. `3 * nf + 6 * nw`
     *  - `njoints`: The number of actuated joints for this stage
     *  - `actuated_joints`: Vector of indices of joints that are actuated
     *  - `unactuated_joints`: Complement vector of `actuated_joints`
     *  - `stage_constraints_name`: Name given to the built constraints object
     *  - `lower_pos_bound`, `upper_pos_bound`: Lower/upper joint position
     *     bounds
     *  - `vel_bound`: Symmetric joint velocity bound
     *  - `tau_bound`: Symmetric joint torque bound
     */
    struct _StageInnateConstraintsContext
    {
        std::vector<std::shared_ptr<Force>> force_data;
        std::vector<std::shared_ptr<Force>> wrench_data;
        uint              nq, nv, nu, nf, nw, force_width, njoints;
        std::vector<uint> actuated_joints, unactuated_joints;
        std::string       stage_constraints_name;
        VecX lower_pos_bound, upper_pos_bound, vel_bound, tau_bound;

        // See the comment on _StageDynamicsContext's equivalent fields; only
        // populated/used when this context backs the ID torque constraints
        // (_buildXMLIDTorqueConstraints), not the plain innate bounds
        // (_buildXMLInnateConstraints)
        mutable std::vector<Vec6>              joint_forces_scratch;
        mutable MatX                           jf_full_scratch;
        mutable _ReducedInverseDynamicsScratch reduced_id_scratch;
    };

    /** Makes the innate constraints for a given stage with context `ctx` based
     * based on the XML, according to the passed templated options
     * Options:
     *  - `FloatingBase`: Whether the system includes a floating base or not
     *  - `IsTerminal`: Whether the stage is the terminal stage (no controls)
     *  - `DM`: The solver mode that should be used
     * 
     * @note
     * If the selected solver mode is InverseDynamics or ReducedInverseDynamics,
     * this function does not generate the required inverse dynamics constraints. 
     * `_makeStageIDTorqueConstraints` needs to also be called.
     */
    template<bool FloatingBase, bool IsTerminal,
             SolverMode SM = SolverMode::ForwardDynamics>
    std::shared_ptr<stagewise::StageConstraints> _makeStageInnateConstraints(
      const std::shared_ptr<const _StageInnateConstraintsContext> &ctx) const;

    /** Makes the inverse dynamics torque constraints required by the InverseDynamics
     * or ReducedInverseDynamics solver modes, based on the XML effort limits.
     * Options:
     *  - `FloatingBase`: Whether the system includes a floating base or not
     *  - `ReducedInverseDynamics`: Whether the solver mode is ReducedInverseDynamics, such
     *     that zero equality constraints are not added
     *  - `WithForces`: Whether there are additional external forces for this
     *     stage
     * 
     * @note
     * This function will only generate inverse dynamics torque constraints,
     * intended for use with the InverseDynamics mode. This function must be
     * called regardless of the value of the `ignore_innate_constraints` option,
     * since InverseDynamics mode always requires these constraints.
     */
    template<bool FloatingBase, bool ReducedInverseDynamics = false,
             bool WithForces = false>
    std::shared_ptr<stagewise::StageConstraints> _makeStageIDTorqueConstraints(
      const std::shared_ptr<const _StageInnateConstraintsContext> &ctx) const;
};
}
}

#include "src/model/Model.tpp"

#endif
