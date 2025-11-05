#ifndef SSQP_PINOCCHIO_WRAPPER_PINOCCHIO_WRAPPER_HPP
#define SSQP_PINOCCHIO_WRAPPER_PINOCCHIO_WRAPPER_HPP

#include <string>
#include <memory>

#include <Eigen/Core>

namespace ssqp
{
namespace pinocchio_wrapper
{
class ModelImpl;

Eigen::Vector3d logSO3(const Eigen::Matrix3d &mat);

Eigen::Matrix3d JlogSO3(const Eigen::Matrix3d &mat);

std::shared_ptr<ModelImpl> makeImpl();

void buildModelFromUrdf(const std::string         &urdf,
                        std::shared_ptr<ModelImpl> impl);

std::pair<uint, uint> modelDimensions(const std::shared_ptr<ModelImpl> &impl);

std::pair<Eigen::VectorXd, Eigen::VectorXd> modelBounds(
  const std::shared_ptr<ModelImpl> &impl);

Eigen::VectorXd integrate(const std::shared_ptr<ModelImpl> &impl,
                          const Eigen::VectorXd &q, const Eigen::VectorXd &v);

Eigen::VectorXd difference(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd            &q0,
                           const Eigen::VectorXd            &q1);

Eigen::VectorXd aba(const std::shared_ptr<ModelImpl> &impl,
                    const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                    const Eigen::VectorXd &tau);

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                           const Eigen::VectorXd &tau,
                           Eigen::MatrixXd       &aba_partial_dq,
                           Eigen::MatrixXd       &aba_partial_dv,
                           Eigen::MatrixXd       &Minv);

void dIntegrate(const std::shared_ptr<ModelImpl> &impl,
                const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                Eigen::MatrixXd &J, uint arg_pos);

// FK including updateFramePlacement
std::pair<Eigen::VectorXd, Eigen::MatrixXd> forwardKinematics(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const std::string &frame);

// Compute and get frame jacobian
void frameJacobian(const std::shared_ptr<ModelImpl> &impl,
                   const Eigen::VectorXd &q, const std::string &frame,
                   Eigen::MatrixXd &J);
}
}

#endif