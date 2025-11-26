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

void printFramesAndJoints(const std::shared_ptr<ModelImpl> &impl);

std::vector<std::string> getOrderedActuatedJoints(
  const std::shared_ptr<ModelImpl> &impl);

Eigen::VectorXd neutralPose(const std::shared_ptr<ModelImpl> &impl);

std::pair<uint, uint> modelDimensions(const std::shared_ptr<ModelImpl> &impl);

uint modelJoints(const std::shared_ptr<ModelImpl> &impl);

std::pair<Eigen::VectorXd, Eigen::VectorXd> modelBounds(
  const std::shared_ptr<ModelImpl> &impl, const bool free_flyer = false);

Eigen::VectorXd integrate(const std::shared_ptr<ModelImpl> &impl,
                          const Eigen::VectorXd &q, const Eigen::VectorXd &v);

Eigen::VectorXd difference(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd            &q0,
                           const Eigen::VectorXd            &q1);

void getFramejMfMatrix(const std::shared_ptr<ModelImpl> &impl,
                       const std::string &frame, uint &joint_id,
                       Eigen::Matrix3d &rotation, Eigen::Vector3d &translation);

Eigen::Vector3d framejMfAct(const std::shared_ptr<ModelImpl> &impl,
                            const std::string                &frame,
                            const Eigen::Vector3d &force, uint &joint_id);

Eigen::VectorXd aba(const std::shared_ptr<ModelImpl> &impl,
                    const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                    const Eigen::VectorXd &tau);

Eigen::VectorXd aba(const std::shared_ptr<ModelImpl> &impl,
                    const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                    const Eigen::VectorXd              &tau,
                    const std::vector<Eigen::Vector3d> &fext);

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                           const Eigen::VectorXd &tau,
                           Eigen::MatrixXd       &aba_partial_dq,
                           Eigen::MatrixXd       &aba_partial_dv,
                           Eigen::MatrixXd       &Minv);

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                           const Eigen::VectorXd              &tau,
                           const std::vector<Eigen::Vector3d> &fext,
                           Eigen::MatrixXd                    &aba_partial_dq,
                           Eigen::MatrixXd                    &aba_partial_dv,
                           Eigen::MatrixXd                    &Minv);

void dIntegrate(const std::shared_ptr<ModelImpl> &impl,
                const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                Eigen::MatrixXd &J, uint arg_pos);

void updateFramePlacements(const std::shared_ptr<ModelImpl> &impl);

// FK including updateFramePlacement
std::pair<Eigen::VectorXd, Eigen::MatrixXd> forwardKinematics(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const std::string &frame);

// FK including updateFramePlacement, returning SE3 matrix
Eigen::MatrixXd forwardKinematicsSE3(const std::shared_ptr<ModelImpl> &impl,
                                     const Eigen::VectorXd            &q,
                                     const std::string                &frame);

std::pair<Eigen::VectorXd, Eigen::MatrixXd> forwardKinematics(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const Eigen::VectorXd &v, const std::string &frame);

Eigen::Vector3d frameVelocity(const std::shared_ptr<ModelImpl> &impl,
                              const std::string &frame, bool local = false);

void frameVelocityDerivatives(const std::shared_ptr<ModelImpl> &impl,
                              Eigen::VectorXd &q, Eigen::VectorXd &v,
                              Eigen::VectorXd &a, const std::string frame,
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

}
}

#endif