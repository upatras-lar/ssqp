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
    Jlog.setZero();
    pinocchio::Jlog6(pinocchio::SE3(mat), Jlog);
    return Jlog;
}

Eigen::Matrix<double, 6, 6> SE3ToActionMatrixInverse(
  const Eigen::Matrix<double, 4, 4> &mat)
{
    return pinocchio::SE3(mat).toActionMatrixInverse();
}

std::shared_ptr<ModelImpl> makeImpl()
{
    return std::make_shared<ModelImpl>();
}

void buildModelFromUrdf(const std::string         &urdf,
                        std::shared_ptr<ModelImpl> impl, bool free_flyer)
{
    if (free_flyer)
    {
        pinocchio::urdf::buildModel(urdf,
                                    pinocchio::JointModelFreeFlyer(),
                                    impl->model);
    }
    else
    {
        pinocchio::urdf::buildModel(urdf, impl->model);
    }
    impl->data = pinocchio::Data(impl->model);
}

std::pair<uint, uint> modelDimensions(const std::shared_ptr<ModelImpl> &impl)
{
    return { impl->model.nq, impl->model.nv };
}

uint modelJoints(const std::shared_ptr<ModelImpl> &impl)
{
    return impl->model.njoints;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> modelBounds(
  const std::shared_ptr<ModelImpl> &impl, const bool free_flyer)
{
    uint            nbounds = impl->model.nq + 2 * impl->model.nv;
    Eigen::VectorXd lb, ub;

    if (free_flyer)
    {
        uint nq = impl->model.nq - 7;
        uint nv = impl->model.nv - 6;

        lb.resize(nbounds - 19);
        ub.resize(nbounds - 19);

        lb.head(nq)        = impl->model.lowerPositionLimit.tail(nq);
        lb.segment(nq, nv) = -impl->model.velocityLimit.tail(nv);
        lb.tail(nv)        = -impl->model.effortLimit.tail(nv);

        ub.head(nq)        = impl->model.upperPositionLimit.tail(nq);
        ub.segment(nq, nv) = impl->model.velocityLimit.tail(nv);
        ub.tail(nv)        = impl->model.effortLimit.tail(nv);
    }
    else
    {
        lb.resize(nbounds);
        ub.resize(nbounds);
        lb << impl->model.lowerPositionLimit, -impl->model.velocityLimit,
          -impl->model.effortLimit;
        ub << impl->model.upperPositionLimit, impl->model.velocityLimit,
          impl->model.effortLimit;
    }

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

void getFramejMfMatrix(const std::shared_ptr<ModelImpl> &impl,
                       const std::string &frame, uint &joint_id,
                       Eigen::Matrix3d &rotation, Eigen::Vector3d &translation)
{
    pinocchio::FrameIndex frame_id = impl->model.getFrameId(frame);
    pinocchio::SE3        jMf      = impl->model.frames[frame_id].placement;
    joint_id                       = impl->model.frames[frame_id].parentJoint;
    rotation                       = jMf.rotation();
    translation                    = jMf.translation();
}

Eigen::Vector3d framejMfAct(const std::shared_ptr<ModelImpl> &impl,
                            const std::string                &frame,
                            const Eigen::Vector3d &force, uint &joint_id)
{
    joint_id           = impl->model.getFrameId(frame);
    pinocchio::Force f = pinocchio::Force::Zero();
    f.linear()         = force;

    pinocchio::Force res = impl->model.frames[joint_id].placement.act(f);

    return res.linear();
}

Eigen::VectorXd aba(const std::shared_ptr<ModelImpl> &impl,
                    const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                    const Eigen::VectorXd              &tau,
                    const std::vector<Eigen::Vector3d> &fext)
{
    pinocchio::container::aligned_vector<pinocchio::Force> forces(
      fext.size(),
      pinocchio::Force::Zero());

    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear() = fext[i];
    }

    pinocchio::aba(impl->model, impl->data, q, v, tau, forces);
    return impl->data.ddq;
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

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                           const Eigen::VectorXd              &tau,
                           const std::vector<Eigen::Vector3d> &fext,
                           Eigen::MatrixXd                    &aba_partial_dq,
                           Eigen::MatrixXd                    &aba_partial_dv,
                           Eigen::MatrixXd                    &Minv)
{
    pinocchio::container::aligned_vector<pinocchio::Force> forces(
      fext.size(),
      pinocchio::Force::Zero());

    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear() = fext[i];
    }

    pinocchio::computeABADerivatives(impl->model,
                                     impl->data,
                                     q,
                                     v,
                                     tau,
                                     forces);
    aba_partial_dq = impl->data.ddq_dq;
    aba_partial_dv = impl->data.ddq_dv;
    Minv           = impl->data.Minv;
}

void printFramesAndJoints(const std::shared_ptr<ModelImpl> &impl)
{
    std::cout << "Number of joints: " << impl->model.njoints << std::endl;
    for (pinocchio::JointIndex i = 0; i < impl->model.njoints; ++i)
    {
        std::cout << "[" << i << "] " << impl->model.names[i] << std::endl;
    }

    std::cout << "\nNumber of frames: " << impl->model.nframes << std::endl;
    for (pinocchio::FrameIndex i = 0; i < impl->model.nframes; ++i)
    {
        const pinocchio::Frame &frame = impl->model.frames[i];
        std::cout << "[" << i << "] " << frame.name << "  type: " << frame.type
                  << std::endl;
    }
}

Eigen::VectorXd neutralPose(const std::shared_ptr<ModelImpl> &impl)
{
    return pinocchio::neutral(impl->model);
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
                   Eigen::MatrixXd &J, bool local)
{
    pinocchio::computeJointJacobians(impl->model, impl->data, q);
    pinocchio::updateFramePlacements(impl->model, impl->data);
    int frame_id = impl->model.getFrameId(frame);
    pinocchio::getFrameJacobian(
      impl->model,
      impl->data,
      frame_id,
      local ? pinocchio::ReferenceFrame::LOCAL
            : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J);
}

void computeJointJacobians(const std::shared_ptr<ModelImpl> &impl,
                           const Eigen::VectorXd            &q)
{
    pinocchio::computeJointJacobians(impl->model, impl->data, q);
    pinocchio::updateFramePlacements(impl->model, impl->data);
}

void frameJacobian(const std::shared_ptr<ModelImpl> &impl,
                   const std::string &frame, Eigen::MatrixXd &J, bool local)
{
    int frame_id = impl->model.getFrameId(frame);
    pinocchio::getFrameJacobian(
      impl->model,
      impl->data,
      frame_id,
      local ? pinocchio::ReferenceFrame::LOCAL
            : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J);
}

}
}