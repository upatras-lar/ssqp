#ifndef SSQP_PINOCCHIO_WRAPPER_PINOCCHIO_WRAPPER_HPP
#define SSQP_PINOCCHIO_WRAPPER_PINOCCHIO_WRAPPER_HPP

#include <string>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "types.hpp"

namespace ssqp
{
namespace pinocchio_wrapper
{
class ModelImpl;

// Whether two vectors hold the same values, compared bitwise
bool sameBits(const VecX &a, const VecX &b);

// Results of a stage's dynamics at one point, so that queries at the same
// point share a single computation
// The model fills and reads it, since the results depend on the stage's forces
// and the actuated joints
struct DynamicsCache
{
    // Point the flags below hold results for
    VecX x, u;
    // The model owns the forces for as long as it has caches
    std::vector<const model::Force *> forces, wrenches;
    // Ignore flags of the forces, then of the wrenches
    std::vector<bool> ignored;

    // Results for the point
    bool joint_forces             = false;
    bool force_jacobian           = false;
    bool acceleration             = false;
    bool torque                   = false;
    bool torque_derivatives       = false;
    bool acceleration_derivatives = false;

    // Drops every result
    void clear()
    {
        joint_forces             = false;
        force_jacobian           = false;
        acceleration             = false;
        torque                   = false;
        torque_derivatives       = false;
        acceleration_derivatives = false;
    }

    // Joint forces and force Jacobian
    std::vector<Vec6> fext;
    MatX              Jf;

    // Acceleration, along with the mass matrix parts, factorization and bias
    // torque ReducedInverseDynamics uses
    VecX              ddq;
    MatX              M, Muu, Muj;
    Eigen::LDLT<MatX> Muu_ldlt;
    VecX              bias;

    // Torque and its RNEA derivatives
    VecX tau;
    MatX dtau_dq, dtau_dv, dtau_da;

    // Acceleration derivatives, from the ABA derivatives in ForwardDynamics
    MatX ddq_dq, ddq_dv, Minv;
    // and in ReducedInverseDynamics, along with the torque derivatives with
    // respect to the actuated accelerations
    MatX da_dq, da_dv, da_daj, da_df, dtau_daj;

    // Temporaries of the ReducedInverseDynamics acceleration derivatives
    MatX Ju;
    MatX dtau_dq_unact, dtau_dv_unact;
    MatX da_dq_reordered, da_dv_reordered, da_daj_reordered, da_df_reordered;
};

Vec3 logSO3(const Mat3 &mat);
Mat3 JlogSO3(const Mat3 &mat);

Vec6 logSE3(const Mat4 &mat);
Mat6 JlogSE3(const Mat4 &mat);

Mat6 SE3ToActionMatrixInverse(const Mat4 &mat);

std::shared_ptr<ModelImpl> makeImpl();

// Kinematics queries keep their results in a kinematics cache and dynamics
// queries in a dynamics cache, so that queries at the same point share a
// single pass
// Dynamics passes run on the active kinematics cache too, so the kinematics
// results they compute along the way are shared as well
// Creates a default cache of each kind at index 0 and one per stage after it,
// so stage k uses index k + 1
void makeStageCaches(const std::shared_ptr<ModelImpl> &impl, const uint stages);

// Index of the kinematics and dynamics caches that queries use
uint activeCacheIndex(const std::shared_ptr<ModelImpl> &impl);
void setActiveCacheIndex(const std::shared_ptr<ModelImpl> &impl,
                         const uint                        index);

// The dynamics cache at the active index
DynamicsCache &activeDynamicsCache(const std::shared_ptr<ModelImpl> &impl);

void buildModelFromUrdf(const std::string         &urdf,
                        std::shared_ptr<ModelImpl> impl,
                        bool                       free_flyer = false);

// Same as buildModelFromUrdf, but `urdf` is the URDF's XML content rather
// than a path to it (used to build models from presets)
void buildModelFromUrdfXML(const std::string         &urdf,
                           std::shared_ptr<ModelImpl> impl,
                           bool                       free_flyer = false);

void buildModelFromMjcf(const std::string         &mjcf,
                        std::shared_ptr<ModelImpl> impl,
                        bool                       free_flyer = false);

bool hasFrame(const std::shared_ptr<ModelImpl> &impl, const std::string &frame);

// Resolves a frame name to its index once; getFrameId() itself does a linear
// (twice-over, to check uniqueness) scan of every frame, so hot-path callers
// should resolve once via this and use the ID-taking overloads below instead
// of repeatedly passing the name
uint getFrameId(const std::shared_ptr<ModelImpl> &impl,
                const std::string                &frame);

void printFramesAndJoints(const std::shared_ptr<ModelImpl> &impl);

std::vector<std::string> getOrderedBodyFrames(
  const std::shared_ptr<ModelImpl> &impl);

enum JointType
{
    Revolute,
    Continuous,
    Prismatic,
    Spherical,
    SphericalEuler,
    Unknown
};

std::vector<JointType> getOrderedJointTypes(
  const std::shared_ptr<ModelImpl> &impl);

std::vector<std::string> getOrderedActuatedJoints(
  const std::shared_ptr<ModelImpl> &impl,
  const std::vector<uint>          &actuated_joints = {});

uint getTotalActuatedJoints(const std::shared_ptr<ModelImpl> &impl);

VecX neutralPose(const std::shared_ptr<ModelImpl> &impl);

std::pair<uint, uint> modelDimensions(const std::shared_ptr<ModelImpl> &impl);

uint modelJoints(const std::shared_ptr<ModelImpl> &impl);

std::vector<uint> modelBoundedJointIndices(
  const std::shared_ptr<ModelImpl> &impl,
  std::unordered_map<uint, uint>   &indices_to_config,
  std::unordered_map<uint, uint>   &indices_to_tangent,
  const bool                        free_flyer = false);

std::pair<VecX, VecX> modelBounds(
  const std::shared_ptr<ModelImpl>     &impl,
  const std::vector<uint>              &bounded_joint_indices,
  const std::unordered_map<uint, uint> &indices_to_config,
  const bool free_flyer = false, const bool skip_velocities = false,
  const bool skip_torques = false);

std::pair<VecX, VecX> modelEffortLimits(const std::shared_ptr<ModelImpl> &impl);

VecX integrate(const std::shared_ptr<ModelImpl> &impl, const VecX &q,
               const VecX &v);

VecX difference(const std::shared_ptr<ModelImpl> &impl, const VecX &q0,
                const VecX &q1);

void getFramejMfMatrix(const std::shared_ptr<ModelImpl> &impl,
                       const std::string &frame, uint &joint_id, Mat3 &rotation,
                       Vec3 &translation);

Vec6 framejMfAct(const std::shared_ptr<ModelImpl> &impl,
                 const std::string &frame, const Vec3 &force, uint &joint_id);

Vec6 framejMfAct(const std::shared_ptr<ModelImpl> &impl, const uint frame_id,
                 const Vec3 &force, uint &joint_id);

// Same as framejMfAct, but for a full spatial wrench (linear force plus
// moment/torque) rather than a pure linear force
Vec6 framejMfActWrench(const std::shared_ptr<ModelImpl> &impl,
                       const uint frame_id, const Vec6 &wrench, uint &joint_id);

// Output-buffer version; resizes `M` as needed
void crba(const std::shared_ptr<ModelImpl> &impl, const VecX &q, MatX &M,
          bool local = false);

VecX aba(const std::shared_ptr<ModelImpl> &impl, const VecX &q, const VecX &v,
         const VecX &tau);

VecX aba(const std::shared_ptr<ModelImpl> &impl, const VecX &q, const VecX &v,
         const VecX &tau, const std::vector<Vec6> &fext);

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const VecX &q, const VecX &v, const VecX &tau,
                           MatX &aba_partial_dq, MatX &aba_partial_dv,
                           MatX &Minv);

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const VecX &q, const VecX &v, const VecX &tau,
                           const std::vector<Vec6> &fext, MatX &aba_partial_dq,
                           MatX &aba_partial_dv, MatX &Minv);

VecX rnea(const std::shared_ptr<ModelImpl> &impl, const VecX &q, const VecX &v,
          const VecX &a);

VecX rnea(const std::shared_ptr<ModelImpl> &impl, const VecX &q, const VecX &v,
          const VecX &a, const std::vector<Vec6> &fext);

void computeRNEADerivatives(const std::shared_ptr<ModelImpl> &impl,
                            const VecX &q, const VecX &v, const VecX &a,
                            MatX &rnea_partial_dq, MatX &rnea_partial_dv,
                            MatX &rnea_partial_da);

void computeRNEADerivatives(const std::shared_ptr<ModelImpl> &impl,
                            const VecX &q, const VecX &v, const VecX &a,
                            const std::vector<Vec6> &fext,
                            MatX &rnea_partial_dq, MatX &rnea_partial_dv,
                            MatX &rnea_partial_da);

void dIntegrate(const std::shared_ptr<ModelImpl> &impl, const VecX &q,
                const VecX &v, MatX &J, const uint arg_pos);

void dDifference(const std::shared_ptr<ModelImpl> &impl, const VecX &q0,
                 const VecX &q1, MatX &J, const uint arg_pos);

// FK including updateFramePlacement
std::pair<Vec3, Mat3> forwardKinematics(const std::shared_ptr<ModelImpl> &impl,
                                        const VecX                       &q,
                                        const std::string &frame);

std::pair<Vec3, Mat3> forwardKinematics(const std::shared_ptr<ModelImpl> &impl,
                                        const VecX &q, const uint frame_id);

// FK including updateFramePlacement, returning SE3 matrix
Mat4 forwardKinematicsSE3(const std::shared_ptr<ModelImpl> &impl, const VecX &q,
                          const std::string &frame);

Mat4 forwardKinematicsSE3(const std::shared_ptr<ModelImpl> &impl, const VecX &q,
                          const uint frame_id);

std::pair<Vec3, Mat3> forwardKinematics(const std::shared_ptr<ModelImpl> &impl,
                                        const VecX &q, const VecX &v,
                                        const std::string &frame);

std::pair<Vec3, Mat3> forwardKinematics(const std::shared_ptr<ModelImpl> &impl,
                                        const VecX &q, const VecX &v,
                                        const uint frame_id);

Vec6 frameVelocity(const std::shared_ptr<ModelImpl> &impl,
                   const std::string &frame, bool local = false);

Vec6 frameVelocity(const std::shared_ptr<ModelImpl> &impl, const uint frame_id,
                   bool local = false);

void frameVelocityDerivatives(const std::shared_ptr<ModelImpl> &impl,
                              const VecX &q, const VecX &v,
                              const std::string frame, Eigen::Ref<MatX> dVdq,
                              Eigen::Ref<MatX> dVdv, bool local = false);

void frameVelocityDerivatives(const std::shared_ptr<ModelImpl> &impl,
                              const VecX &q, const VecX &v, const uint frame_id,
                              Eigen::Ref<MatX> dVdq, Eigen::Ref<MatX> dVdv,
                              bool local = false);

// Compute and get frame jacobian
void frameJacobian(const std::shared_ptr<ModelImpl> &impl, const VecX &q,
                   const std::string &frame, Eigen::Ref<MatX> J,
                   bool local = false);

void frameJacobian(const std::shared_ptr<ModelImpl> &impl, const VecX &q,
                   const uint frame_id, Eigen::Ref<MatX> J, bool local = false);

void computeJointJacobians(const std::shared_ptr<ModelImpl> &impl,
                           const VecX                       &q);

// Get frame jacobian (assumes computeJointJacobians has been called, with no
// dynamics pass at another point since)
void frameJacobian(const std::shared_ptr<ModelImpl> &impl,
                   const std::string &frame, Eigen::Ref<MatX> J,
                   bool local = false);

void frameJacobian(const std::shared_ptr<ModelImpl> &impl, const uint frame_id,
                   Eigen::Ref<MatX> J, bool local = false);

// Get center of mass
Vec3 centerOfMass(const std::shared_ptr<ModelImpl> &impl, const VecX &q);

MatX jacobianCenterOfMass(const std::shared_ptr<ModelImpl> &impl,
                          const VecX                       &q);

VecX randomConfiguration(const std::shared_ptr<ModelImpl> &impl);

VecX getAllDynamicParameters(const std::shared_ptr<ModelImpl> &impl);

void setAllDynamicParameters(std::shared_ptr<ModelImpl> &impl,
                             const VecX                 &parameters);

MatX computeJointTorqueRegressor(const std::shared_ptr<ModelImpl> &impl,
                                 const VecX &q, const VecX &v, const VecX &a);
}
}

#endif