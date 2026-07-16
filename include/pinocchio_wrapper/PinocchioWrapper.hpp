#ifndef SSQP_PINOCCHIO_WRAPPER_PINOCCHIO_WRAPPER_HPP
#define SSQP_PINOCCHIO_WRAPPER_PINOCCHIO_WRAPPER_HPP

#include <string>
#include <memory>
#include <vector>

#include <Eigen/Core>

namespace ssqp
{
namespace pinocchio_wrapper
{
class ModelImpl;

Eigen::Vector3d logSO3(const Eigen::Matrix3d &mat);
Eigen::Matrix3d JlogSO3(const Eigen::Matrix3d &mat);

Eigen::Vector<double, 6>    logSE3(const Eigen::Matrix<double, 4, 4> &mat);
Eigen::Matrix<double, 6, 6> JlogSE3(const Eigen::Matrix<double, 4, 4> &mat);

Eigen::Matrix<double, 6, 6> SE3ToActionMatrixInverse(
  const Eigen::Matrix<double, 4, 4> &mat);

std::shared_ptr<ModelImpl> makeImpl();

void buildModelFromUrdf(const std::string         &urdf,
                        std::shared_ptr<ModelImpl> impl,
                        bool                       free_flyer = false);

void buildModelFromMjcf(const std::string         &mjcf,
                        std::shared_ptr<ModelImpl> impl,
                        bool                       free_flyer = false);

bool hasFrame(const std::shared_ptr<ModelImpl> &impl, const std::string &frame);

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

Eigen::VectorXd neutralPose(const std::shared_ptr<ModelImpl> &impl);

std::pair<uint, uint> modelDimensions(const std::shared_ptr<ModelImpl> &impl);

uint modelJoints(const std::shared_ptr<ModelImpl> &impl);

std::vector<uint> modelBoundedJointIndices(
  const std::shared_ptr<ModelImpl> &impl,
  std::unordered_map<uint, uint>   &indices_to_config,
  std::unordered_map<uint, uint>   &indices_to_tangent,
  const bool                        free_flyer = false);

std::pair<Eigen::VectorXd, Eigen::VectorXd> modelBounds(
  const std::shared_ptr<ModelImpl>     &impl,
  const std::vector<uint>              &bounded_joint_indices,
  const std::unordered_map<uint, uint> &indices_to_config,
  const bool free_flyer = false, const bool skip_velocities = false,
  const bool skip_torques = false);

std::pair<Eigen::VectorXd, Eigen::VectorXd> modelEffortLimits(
  const std::shared_ptr<ModelImpl> &impl);

Eigen::VectorXd integrate(const std::shared_ptr<ModelImpl> &impl,
                          const Eigen::VectorXd &q, const Eigen::VectorXd &v);

Eigen::VectorXd difference(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd            &q0,
                           const Eigen::VectorXd            &q1);

void getFramejMfMatrix(const std::shared_ptr<ModelImpl> &impl,
                       const std::string &frame, uint &joint_id,
                       Eigen::Matrix3d &rotation, Eigen::Vector3d &translation);

Eigen::Vector<double, 6> framejMfAct(const std::shared_ptr<ModelImpl> &impl,
                                     const std::string                &frame,
                                     const Eigen::Vector3d            &force,
                                     uint &joint_id);

Eigen::MatrixXd crba(const std::shared_ptr<ModelImpl> &impl,
                     const Eigen::VectorXd &q, bool local = false);

Eigen::VectorXd aba(const std::shared_ptr<ModelImpl> &impl,
                    const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                    const Eigen::VectorXd &tau);

Eigen::VectorXd aba(const std::shared_ptr<ModelImpl> &impl,
                    const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                    const Eigen::VectorXd                       &tau,
                    const std::vector<Eigen::Vector<double, 6>> &fext);

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                           const Eigen::VectorXd &tau,
                           Eigen::MatrixXd       &aba_partial_dq,
                           Eigen::MatrixXd       &aba_partial_dv,
                           Eigen::MatrixXd       &Minv);

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                           const Eigen::VectorXd                       &tau,
                           const std::vector<Eigen::Vector<double, 6>> &fext,
                           Eigen::MatrixXd &aba_partial_dq,
                           Eigen::MatrixXd &aba_partial_dv,
                           Eigen::MatrixXd &Minv);

Eigen::VectorXd rnea(const std::shared_ptr<ModelImpl> &impl,
                     const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                     const Eigen::VectorXd &a);

Eigen::VectorXd rnea(const std::shared_ptr<ModelImpl> &impl,
                     const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                     const Eigen::VectorXd                       &a,
                     const std::vector<Eigen::Vector<double, 6>> &fext);

void computeRNEADerivatives(const std::shared_ptr<ModelImpl> &impl,
                            const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                            const Eigen::VectorXd &a,
                            Eigen::MatrixXd       &rnea_partial_dq,
                            Eigen::MatrixXd       &rnea_partial_dv,
                            Eigen::MatrixXd       &rnea_partial_da);

void computeRNEADerivatives(const std::shared_ptr<ModelImpl> &impl,
                            const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                            const Eigen::VectorXd                       &a,
                            const std::vector<Eigen::Vector<double, 6>> &fext,
                            Eigen::MatrixXd &rnea_partial_dq,
                            Eigen::MatrixXd &rnea_partial_dv,
                            Eigen::MatrixXd &rnea_partial_da);

void dIntegrate(const std::shared_ptr<ModelImpl> &impl,
                const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                Eigen::MatrixXd &J, const uint arg_pos);

void dDifference(const std::shared_ptr<ModelImpl> &impl,
                 const Eigen::VectorXd &q0, const Eigen::VectorXd &q1,
                 Eigen::MatrixXd &J, const uint arg_pos);

void updateFramePlacements(const std::shared_ptr<ModelImpl> &impl);

// FK including updateFramePlacement
std::pair<Eigen::Vector3d, Eigen::Matrix3d> forwardKinematics(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const std::string &frame);

// FK including updateFramePlacement, returning SE3 matrix
Eigen::Matrix4d forwardKinematicsSE3(const std::shared_ptr<ModelImpl> &impl,
                                     const Eigen::VectorXd            &q,
                                     const std::string                &frame);

std::pair<Eigen::Vector3d, Eigen::Matrix3d> forwardKinematics(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const Eigen::VectorXd &v, const std::string &frame);

Eigen::Vector<double, 6> frameVelocity(const std::shared_ptr<ModelImpl> &impl,
                                       const std::string                &frame,
                                       bool local = false);

void frameVelocityDerivatives(const std::shared_ptr<ModelImpl> &impl,
                              const Eigen::VectorXd            &q,
                              const Eigen::VectorXd &v, const std::string frame,
                              Eigen::MatrixXd &dVdq, Eigen::MatrixXd &dVdv,
                              bool local = false);

// Compute and get frame jacobian
void frameJacobian(const std::shared_ptr<ModelImpl> &impl,
                   const Eigen::VectorXd &q, const std::string &frame,
                   Eigen::MatrixXd &J, bool local = false);

void computeJointJacobians(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd            &q);

// Get frame jacobian (assumes computeJointJacobians has been called)
void frameJacobian(const std::shared_ptr<ModelImpl> &impl,
                   const std::string &frame, Eigen::MatrixXd &J,
                   bool local = false);

// Get center of mass
Eigen::Vector3d centerOfMass(const std::shared_ptr<ModelImpl> &impl,
                             const Eigen::VectorXd            &q);

Eigen::MatrixXd jacobianCenterOfMass(const std::shared_ptr<ModelImpl> &impl,
                                     const Eigen::VectorXd            &q);

Eigen::VectorXd randomConfiguration(const std::shared_ptr<ModelImpl> &impl);

Eigen::VectorXd getAllDynamicParameters(const std::shared_ptr<ModelImpl> &impl);

void setAllDynamicParameters(std::shared_ptr<ModelImpl> &impl,
                             const Eigen::VectorXd      &parameters);

Eigen::MatrixXd computeJointTorqueRegressor(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const Eigen::VectorXd &v, const Eigen::VectorXd &a);
}
}

#endif