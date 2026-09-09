#ifndef SSQP_MODEL_MODEL_HPP
#define SSQP_MODEL_MODEL_HPP

#include <unordered_set>
#include <variant>
#include <type_traits>
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

    // Add a new running cost of StageCost type L
    // A running cost is a cost added to every non-terminal stage
    template<typename L, typename... Args>
    std::shared_ptr<stagewise::StageCost> addRunningCost(Args... args);

    // Add a new running cost from existing object
    // A running cost is a cost added to every non-terminal stage
    void addRunningCost(const std::shared_ptr<stagewise::StageCost> &cost);

    // Add new constraints for stage k from existing object
    void addConstraints(
      const uint                                          k,
      const std::shared_ptr<stagewise::StageConstraints> &constraints);

    // Add a new external force as an additional decision variable for stage k
    // on the given frame
    // Returns the newly created force
    std::shared_ptr<Force> addForce(const uint k, const std::string &frame);

    // Returns the number of external forces added for a given stage
    uint nforces(const uint k) const;

    // Vector containing frame names, ordered 1-1 with forces for stage k
    std::vector<std::string> orderedForceFrames(const uint k) const;

    // Whether the model has a frame with the given name
    bool hasFrame(const std::string &frame) const;

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

    const Mat4 forwardKinematicsSE3(const VecX        &q,
                                    const std::string &frame) const;

    // Get the velocity for a given frame
    const Vec6 frameVelocity(const VecX &q, const VecX &v,
                             const std::string &frame,
                             bool               local = false) const;

    std::pair<MatX, MatX> frameVelocityDerivatives(const VecX &q, const VecX &v,
                                                   const std::string &frame,
                                                   bool local = false) const;

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

    // Get frame jacobian (assuming computeJointJacobians has been called)
    const MatX frameJacobian(const std::string &frame,
                             bool               local = false) const;

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
    std::vector<std::vector<std::shared_ptr<Force>>> _forces;

    // Optional custom properties for non-XML models
    bool             _has_custom_properties = false;
    CustomProperties _custom_properties {};

    // Helper functions

    /** Calculates joint forces given a configuration and an external force
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
      const VecX &q, const VecX &f,
      const std::vector<std::shared_ptr<Force>> &force_data) const;

    /** Calculates joint forces and total force jacobian given a configuration
     * and an external force vector
     *
     * Same as _calculateJointForces, but also calculates the full force
     * jacobian
     */
    std::pair<std::vector<Vec6>, MatX> _calculateJointForcesAndJacobian(
      const VecX &q, const VecX &f,
      const std::vector<std::shared_ptr<Force>> &force_data) const;

    /** Calculates the full acceleration vector for ReducedInverseDynamics mode
     * Input:
     *  - `q`: The configuration vector
     *  - `v`: The velocity vector
     *  - `ctrl`: The non-force part of the control vector; this is assumed to
     *     be the accelerations for each of the actuated joints
     *  - `f`: The force part of the control vector
     *  - `force_data`: Vector of data about each force in `f`
     *  - `actuated_joints`: Vector of indices of joints that are actuated
     *  - `unactuated_joints`: Complement vector of `actuated_joints`
     * 
     * Output:
     *  - `Muu`: The unactuated part of the mass matrix (reordered)
     *  - `Muj`: The top-right part of the reordered mass matrix 
     *     (unactuated-actuated)
     *  - `Muu_ldlt`: Eigen LDLT object for LDLT decomposition of `Muu`
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
      const std::vector<uint>                   &actuated_joints,
      const std::vector<uint> &unactuated_joints, MatX &Muu, MatX &Muj,
      Eigen::LDLT<MatX> &Muu_ldlt, std::vector<Vec6> &joint_forces,
      MatX &Jf_full) const;

    /** Helper that calls `_calculateReducedInverseFullAcceleration` and ignores
     * output arguments
     */
    template<bool FloatingBase, bool WithForces>
    VecX _calculateReducedInverseFullAcceleration(
      const VecX &q, const VecX &v, const VecX &ctrl, const VecX &f,
      const std::vector<std::shared_ptr<Force>> &force_data,
      const std::vector<uint>                   &actuated_joints,
      const std::vector<uint>                   &unactuated_joints) const;

    /** Calculates the derivatives of the acceleration vector for ReducedInverseDynamics
     * mode, with respect to q, v, aj (the actual actuated controls) and f
     * Input:
     *  - `q`: The configuration vector
     *  - `v`: The velocity vector
     *  - `ctrl`: The non-force part of the control vector; this is assumed to
     *     be the accelerations for each of the actuated joints
     *  - `f`: The force part of the control vector
     *  - `force_data`: Vector of data about each force in `f`
     *  - `actuated_joints`: Vector of indices of joints that are actuated
     *  - `unactuated_joints`: Complement vector of `actuated_joints`
     * 
     * Inputs from `_calculateReducedInverseFullAcceleration`'s output:
     *  - `ddq`: The full acceleration vector
     *  - `Muu`: The unactuated part of the mass matrix (reordered)
     *  - `Muj`: The top-right part of the reordered mass matrix 
     *     (unactuated-actuated)
     *  - `Muu_ldlt`: Eigen LDLT object for LDLT decomposition of `Muu`
     *  - `joint_forces`: The forces on each joint
     *  - `Jf_full`: The full force jacobian
     * 
     * Output:
     *  - `dtau_dq`: The RNEA derivative with respect to the configuration (q)
     *  - `dtau_dv`: The RNEA derivative with respect to the velocity (v)
     *  - `dtau_da`: The RNEA derivative with respect to the full acceleration
     *     (a)
     *  - `dtau_daj`: The RNEA derivative with respect to the actuated part of
     *     the acceleration (aj)
     *  - `da_dq`: The derivative of the full acceleration vector with respect
     *     to the configuration (q)
     *  - `da_dv`: The derivative of the full acceleration vector with respect
     *     to the velocity (v)
     *  - `da_daj`: The derivative of the full acceleration vector with respect
     *     to the actuated part of the acceleration
     *  - `da_df`: The derivative of the full acceleration vector with respect
     *     to the forces
     */
    template<bool FloatingBase, bool WithForces>
    void _calculateReducedInverseFullAccelerationDerivatives(
      const VecX &q, const VecX &v, const VecX &ctrl, const VecX &f,
      const std::vector<std::shared_ptr<Force>> &force_data,
      const std::vector<uint>                   &actuated_joints,
      const std::vector<uint> &unactuated_joints, const VecX &ddq,
      const MatX &Muu, const MatX &Muj, const Eigen::LDLT<MatX> &Muu_ldlt,
      const std::vector<Vec6> &joint_forces, const MatX &Jf_full, MatX &dtau_dq,
      MatX &dtau_dv, MatX &dtau_da, MatX &dtau_daj, MatX &da_dq, MatX &da_dv,
      MatX &da_daj, MatX &da_df) const;

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
     *  - `S`: Selection matrix for stage ($S \tau$)
     *  - `force_data`: Vector containing a Force object for each force
     *  - `init_state`: Initial state for stage
     *  - `init_control`: Initial controls for stage
     *  - `nq`: The configuration size for this stage
     *  - `nv`: The velocity size for this stage
     *  - `nu`: The size of the non-force part of the controls for this stage
     *  - `nf`: The size of the force part of the controls for this stage
     *  - `actuated_joints`: Vector of indices of joints that are actuated
     *  - `unactuated_joints`: Complement vector of `actuated_joints`
     *  - `floating_base`: Whether the model has a floating base for this stage
     */
    struct _StageDynamicsContext
    {
        const MatX                          S;
        std::vector<std::shared_ptr<Force>> force_data;
        VecX                                init_state;
        VecX                                init_control;
        uint                                nq, nv, nu, nf;
        std::vector<uint>                   actuated_joints, unactuated_joints;
        std::string                         stage_dynamics_name;
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
      const _StageDynamicsContext &ctx) const;

    /** Context for `_makeStageConstraints` for a given stage
     * Data:
     *  - `force_data`: Vector containing a Force object for each force
     */
    struct _StageInnateConstraintsContext
    {
        std::vector<std::shared_ptr<Force>> force_data;
        uint                                nq, nv, nu, nf, njoints;
        std::vector<uint>                   actuated_joints, unactuated_joints;
        std::string                         stage_constraints_name;
        VecX lower_pos_bound, upper_pos_bound, vel_bound, tau_bound;
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
      const _StageInnateConstraintsContext &ctx) const;

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
      const _StageInnateConstraintsContext &ctx) const;
};
}
}

#include "src/model/Model.tpp"

#endif
