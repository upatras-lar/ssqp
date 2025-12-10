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
     *  - `dynamics_mode`: The dynamics mode to use (default Inverse Dynamics)
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
        DynamicsMode                      dynamics_mode = Inverse;
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
     * finalize it with the default parameters.
     */
    Model(uint horizon, double dt,
          const Structure &structure = Structure::Default());

    /* Model Config Constructor
     * ---
     * Initializes model with default structure and the given config
     * 
     * Input:
     *  - `horizon`: The time horizon to optimize over
     *  - `dt`: Time step per solver stage
     *  - `config`: The system configuration
     * 
     * #### Note:
     * This constructor finalizes the model.
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
     * the default parameters will be used.
     */
    void finalize(const Config &config = Config::Default());

    // Whether the model is finalized
    bool finalized() const;

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

    // Add new constraints for stage k of StageConstraint type L
    template<typename C, typename... Args>
    void addConstraints(const uint k, Args... args);

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

    void printFramesAndJoints();

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

    const VecX inverseDynamics(
      const VecX &q, const VecX &v, const VecX &a,
      const std::unordered_map<std::string, std::vector<Vec3>> &forces) const;

    // Get a frame jacobian based on the URDF model dynamics
    const MatX frameJacobian(const VecX &x, const std::string &frame,
                             bool local = false) const;

    // Get frame jacobian (assuming computeJointJacobians has been called)
    const MatX frameJacobian(const std::string &frame,
                             bool               local = false) const;

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
    CostsContainer                  _costs;
    DynamicsContainer               _dynamics;
    ConstraintsContainer            _constraints;
    std::vector<std::vector<Force>> _forces;

    // Functionality functions
    void _addURDFInnateConstraints();
    void _buildURDFDynamics(const Config &config);

    VecX _randomVector(const uint size, const double start = -5,
                       const double end = 5) const;

    struct _StageDynamicsContext
    {
        const MatX         S;
        const uint         total_joints;
        uint               nf;
        std::vector<Force> forces;
        VecX               init_state;
        VecX               init_control;
    };

    template<bool WithForces, IntegrationScheme IS>
    std::shared_ptr<stagewise::StageDynamics> _makeStageDynamics(
      const _StageDynamicsContext &ctx) const;

    template<bool FloatingBase>
    std::pair<std::shared_ptr<stagewise::StageConstraints>,
              std::shared_ptr<stagewise::StageConstraints>>
    _makeInnateConstraints() const;
};
}
}

#include "src/model/Model.tpp"

#endif