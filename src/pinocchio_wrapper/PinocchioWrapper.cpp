#include "pinocchio_wrapper/PinocchioWrapper.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace ssqp
{
namespace pinocchio_wrapper
{
struct ModelImpl
{
    pinocchio::Model model;
    pinocchio::Data  data;
};

Eigen::Vector3d logSO3(const Eigen::Matrix3d &mat)
{
    return pinocchio::log3(mat);
}

Eigen::Matrix3d JlogSO3(const Eigen::Matrix3d &mat)
{
    Eigen::Matrix3d Jlog;
    pinocchio::Jlog3(mat, Jlog);
    return Jlog;
}

Eigen::Vector<double, 6> logSE3(const Eigen::Matrix<double, 4, 4> &mat)
{
    return pinocchio::log6(mat);
}

Eigen::Matrix<double, 6, 6> JlogSE3(const Eigen::Matrix<double, 4, 4> &mat)
{
    Eigen::Matrix<double, 6, 6> Jlog;
    pinocchio::Jlog6(pinocchio::SE3(mat));
    return Jlog;
}

std::shared_ptr<ModelImpl> makeImpl()
{
    return std::make_shared<ModelImpl>();
}

void buildModelFromUrdf(const std::string         &urdf,
                        std::shared_ptr<ModelImpl> impl)
{
    pinocchio::urdf::buildModel(urdf, impl->model);
    impl->data = pinocchio::Data(impl->model);
}

std::pair<uint, uint> modelDimensions(const std::shared_ptr<ModelImpl> &impl)
{
    return { impl->model.nq, impl->model.nv };
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> modelBounds(
  const std::shared_ptr<ModelImpl> &impl)
{
    Eigen::VectorXd lb(impl->model.nq + 2 * impl->model.nv),
      ub(impl->model.nq + 2 * impl->model.nv);
    lb << impl->model.lowerPositionLimit, -impl->model.velocityLimit,
      -impl->model.effortLimit;
    ub << impl->model.upperPositionLimit, impl->model.velocityLimit,
      impl->model.effortLimit;

    return { lb, ub };
}

Eigen::VectorXd integrate(const std::shared_ptr<ModelImpl> &impl,
                          const Eigen::VectorXd &q, const Eigen::VectorXd &v)
{
    return pinocchio::integrate(impl->model, q, v);
}

Eigen::VectorXd difference(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd &q0, const Eigen::VectorXd &q1)
{
    return pinocchio::difference(impl->model, q0, q1);
}

Eigen::VectorXd aba(const std::shared_ptr<ModelImpl> &impl,
                    const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                    const Eigen::VectorXd &tau)
{
    pinocchio::aba(impl->model, impl->data, q, v, tau);
    return impl->data.ddq;
}

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                           const Eigen::VectorXd &tau,
                           Eigen::MatrixXd       &aba_partial_dq,
                           Eigen::MatrixXd       &aba_partial_dv,
                           Eigen::MatrixXd       &Minv)
{
    pinocchio::computeABADerivatives(impl->model, impl->data, q, v, tau);
    aba_partial_dq = impl->data.ddq_dq;
    aba_partial_dv = impl->data.ddq_dv;
    Minv           = impl->data.Minv;
}

void dIntegrate(const std::shared_ptr<ModelImpl> &impl,
                const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                Eigen::MatrixXd &J, uint arg_pos)
{
    pinocchio::dIntegrate(impl->model,
                          q,
                          v,
                          J,
                          static_cast<pinocchio::ArgumentPosition>(arg_pos));
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> forwardKinematics(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const std::string &frame)
{
    pinocchio::forwardKinematics(impl->model, impl->data, q);
    int frame_id = impl->model.getFrameId(frame);
    pinocchio::updateFramePlacement(impl->model, impl->data, frame_id);

    return { impl->data.oMf[frame_id].translation(),
             impl->data.oMf[frame_id].rotation() };
}

Eigen::MatrixXd forwardKinematicsSE3(const std::shared_ptr<ModelImpl> &impl,
                                     const Eigen::VectorXd            &q,
                                     const std::string                &frame)
{
    pinocchio::forwardKinematics(impl->model, impl->data, q);
    int frame_id = impl->model.getFrameId(frame);
    pinocchio::updateFramePlacement(impl->model, impl->data, frame_id);

    return impl->data.oMf[frame_id].toHomogeneousMatrix();
}

void frameJacobian(const std::shared_ptr<ModelImpl> &impl,
                   const Eigen::VectorXd &q, const std::string &frame,
                   Eigen::MatrixXd &J)
{
    pinocchio::computeJointJacobians(impl->model, impl->data, q);
    int frame_id = impl->model.getFrameId(frame);
    pinocchio::getFrameJacobian(impl->model,
                                impl->data,
                                frame_id,
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                J);
}
}
}