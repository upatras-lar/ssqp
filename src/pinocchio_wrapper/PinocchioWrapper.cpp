#include "pinocchio_wrapper/PinocchioWrapper.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>

#include <boost/variant.hpp>

namespace ssqp
{
namespace pinocchio_wrapper
{
struct ModelImpl
{
    pinocchio::Model model;
    pinocchio::Data  data;
};

struct JointTypeIdentifier : public boost::static_visitor<void>
{
    const pinocchio::Model &model;
    pinocchio::JointIndex   joint_id;

    JointTypeIdentifier(const pinocchio::Model &model, pinocchio::JointIndex id)
      : model(model)
      , joint_id(id)
    {
    }
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

void buildModelFromMjcf(const std::string         &mjcf,
                        std::shared_ptr<ModelImpl> impl, bool free_flyer)
{
    if (free_flyer)
    {
        pinocchio::mjcf::buildModel(mjcf,
                                    pinocchio::JointModelFreeFlyer(),
                                    impl->model);
    }
    else
    {
        pinocchio::mjcf::buildModel(mjcf, impl->model);
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

std::vector<uint> modelBoundedJointIndices(
  const std::shared_ptr<ModelImpl> &impl,
  std::unordered_map<uint, uint>   &indices_to_config,
  std::unordered_map<uint, uint> &indices_to_tangent, const bool free_flyer)
{
    uint offset = 1;
    if (free_flyer)
    {
        offset += 1;
    }

    std::vector<uint> indices;

    uint acc_config  = 0;
    uint acc_tangent = 0;

    for (pinocchio::JointIndex i = 1; i < impl->model.njoints; ++i)
    {
        const std::string jname = impl->model.joints[i].shortname();

        if ((jname.find("JointModelR") != std::string::npos &&
             impl->model.joints[i].nq() == 1) // Revolute Joint with bounds
            || (jname.find("JointModelP") != std::string::npos &&
                impl->model.joints[i].nq() == 1)) // Prismatic Joint

        {
            indices.push_back(i - offset);
            indices_to_config[i - offset]  = acc_config;
            indices_to_tangent[i - offset] = acc_tangent;
        }

        acc_config  += impl->model.joints[i].nq();
        acc_tangent += impl->model.joints[i].nv();
    }

    return indices;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> modelBounds(
  const std::shared_ptr<ModelImpl>     &impl,
  const std::vector<uint>              &bounded_joint_indices,
  const std::unordered_map<uint, uint> &indices_to_config,
  const bool free_flyer, const bool skip_velocities, const bool skip_torques)
{
    uint nv_reduced = impl->model.nv;
    if (free_flyer)
    {
        nv_reduced -= 6;
    }

    uint nbounds = bounded_joint_indices.size() +
                   (!skip_velocities + !skip_torques) * nv_reduced;
    Eigen::VectorXd lb(nbounds), ub(nbounds);

    for (uint i = 0; i < bounded_joint_indices.size(); ++i)
    {
        lb[i] =
          impl->model
            .lowerPositionLimit[indices_to_config.at(bounded_joint_indices[i])];
        ub[i] =
          impl->model
            .upperPositionLimit[indices_to_config.at(bounded_joint_indices[i])];
    }

    if (!skip_velocities)
    {
        lb.segment(bounded_joint_indices.size(), nv_reduced) =
          -impl->model.velocityLimit.tail(nv_reduced);
        ub.segment(bounded_joint_indices.size(), nv_reduced) =
          impl->model.velocityLimit.tail(nv_reduced);
    }

    if (!skip_torques)
    {
        lb.segment(bounded_joint_indices.size() + nv_reduced, nv_reduced) =
          -impl->model.effortLimit.tail(nv_reduced);
        ub.segment(bounded_joint_indices.size() + nv_reduced, nv_reduced) =
          impl->model.effortLimit.tail(nv_reduced);
    }

    return { lb, ub };
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> modelEffortLimits(
  const std::shared_ptr<ModelImpl> &impl)
{
    return { -impl->model.effortLimit, impl->model.effortLimit };
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

Eigen::Vector<double, 6> framejMfAct(const std::shared_ptr<ModelImpl> &impl,
                                     const std::string                &frame,
                                     const Eigen::Vector3d            &force,
                                     uint                             &joint_id)
{
    uint             frame_id = impl->model.getFrameId(frame);
    pinocchio::Force f        = pinocchio::Force::Zero();
    f.linear()                = force;

    pinocchio::Force res = impl->model.frames[frame_id].placement.act(f);
    joint_id             = impl->model.frames[frame_id].parentJoint;

    return res.toVector();
}

Eigen::MatrixXd crba(const std::shared_ptr<ModelImpl> &impl,
                     const Eigen::VectorXd &q, bool local)
{
    pinocchio::crba(impl->model,
                    impl->data,
                    q,
                    local ? pinocchio::Convention::LOCAL
                          : pinocchio::Convention::WORLD);

    impl->data.M = impl->data.M.selfadjointView<Eigen::Upper>();

    return impl->data.M;
}

Eigen::VectorXd aba(const std::shared_ptr<ModelImpl> &impl,
                    const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                    const Eigen::VectorXd                       &tau,
                    const std::vector<Eigen::Vector<double, 6>> &fext)
{
    pinocchio::container::aligned_vector<pinocchio::Force> forces(fext.size());

    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear()  = fext[i].head(3);
        forces[i].angular() = fext[i].tail(3);
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
                           const Eigen::VectorXd                       &tau,
                           const std::vector<Eigen::Vector<double, 6>> &fext,
                           Eigen::MatrixXd &aba_partial_dq,
                           Eigen::MatrixXd &aba_partial_dv,
                           Eigen::MatrixXd &Minv)
{
    pinocchio::container::aligned_vector<pinocchio::Force> forces(fext.size());

    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear()  = fext[i].head(3);
        forces[i].angular() = fext[i].tail(3);
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

Eigen::VectorXd rnea(const std::shared_ptr<ModelImpl> &impl,
                     const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                     const Eigen::VectorXd &a)
{
    return pinocchio::rnea(impl->model, impl->data, q, v, a);
}

Eigen::VectorXd rnea(const std::shared_ptr<ModelImpl> &impl,
                     const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                     const Eigen::VectorXd                       &a,
                     const std::vector<Eigen::Vector<double, 6>> &fext)
{
    pinocchio::container::aligned_vector<pinocchio::Force> forces(
      fext.size(),
      pinocchio::Force::Zero());

    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear()  = fext[i].head<3>();
        forces[i].angular() = fext[i].tail<3>();
    }

    return pinocchio::rnea(impl->model, impl->data, q, v, a, forces);
}

void computeRNEADerivatives(const std::shared_ptr<ModelImpl> &impl,
                            const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                            const Eigen::VectorXd &a,
                            Eigen::MatrixXd       &rnea_partial_dq,
                            Eigen::MatrixXd       &rnea_partial_dv,
                            Eigen::MatrixXd       &rnea_partial_da)
{
    pinocchio::computeRNEADerivatives(impl->model, impl->data, q, v, a);

    rnea_partial_dq = impl->data.dtau_dq;
    rnea_partial_dv = impl->data.dtau_dv;
    rnea_partial_da = impl->data.M;

    rnea_partial_da = rnea_partial_da.selfadjointView<Eigen::Upper>();
}

void computeRNEADerivatives(const std::shared_ptr<ModelImpl> &impl,
                            const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                            const Eigen::VectorXd                       &a,
                            const std::vector<Eigen::Vector<double, 6>> &fext,
                            Eigen::MatrixXd &rnea_partial_dq,
                            Eigen::MatrixXd &rnea_partial_dv,
                            Eigen::MatrixXd &rnea_partial_da)
{
    pinocchio::container::aligned_vector<pinocchio::Force> forces(
      fext.size(),
      pinocchio::Force::Zero());

    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear()  = fext[i].head<3>();
        forces[i].angular() = fext[i].tail<3>();
    }

    pinocchio::computeRNEADerivatives(impl->model, impl->data, q, v, a, forces);

    rnea_partial_dq = impl->data.dtau_dq;
    rnea_partial_dv = impl->data.dtau_dv;
    rnea_partial_da = impl->data.M;

    rnea_partial_da = rnea_partial_da.selfadjointView<Eigen::Upper>();
}

bool hasFrame(const std::shared_ptr<ModelImpl> &impl, const std::string &frame)
{
    for (pinocchio::FrameIndex i = 0; i < impl->model.nframes; ++i)
    {
        const pinocchio::Frame &model_frame = impl->model.frames[i];
        if (frame == model_frame.name)
        {
            return true;
        }
    }

    return false;
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

std::vector<std::string> getOrderedBodyFrames(
  const std::shared_ptr<ModelImpl> &impl)
{
    std::vector<std::string> names;
    names.reserve(impl->model.frames.size());

    for (pinocchio::FrameIndex i = 0; i < impl->model.nframes; ++i)
    {
        const pinocchio::Frame &frame = impl->model.frames[i];

        if (frame.type == pinocchio::FrameType::OP_FRAME ||
            frame.type == pinocchio::FrameType::BODY)
        {
            names.push_back(frame.name);
        }
    }

    return names;
}

std::vector<JointType> getOrderedJointTypes(
  const std::shared_ptr<ModelImpl> &impl)
{
    std::vector<JointType> types;
    types.reserve(impl->model.joints.size());

    for (pinocchio::JointIndex jid = 1; jid < impl->model.joints.size(); ++jid)
    {
        const auto &j = impl->model.joints[jid];

        if (j.nv() == 0) continue;

        std::string shortname = j.shortname();

        // Skip floating-base joints
        if (shortname == "JointModelFreeFlyer" ||
            shortname == "JointModelPlanar" ||
            shortname == "JointModelTranslation")
        {
            continue;
        }

        if (shortname.find("JointModelRUB") != std::string::npos)
        {
            types.push_back(JointType::Continuous);
            continue;
        }

        if (shortname.find("JointModelR") != std::string::npos)
        {
            types.push_back(JointType::Revolute);
            continue;
        }

        if (shortname == "JointModelSpherical")
        {
            types.push_back(JointType::Spherical);
            continue;
        }

        if (shortname == "JointModelSphericalZYX")
        {
            types.push_back(JointType::SphericalEuler);
            continue;
        }

        if (shortname.find("JointModelP") != std::string::npos)
        {
            types.push_back(JointType::Prismatic);
            continue;
        }

        types.push_back(JointType::Unknown);
    }

    return types;
}

std::vector<std::string> getOrderedActuatedJoints(
  const std::shared_ptr<ModelImpl> &impl,
  const std::vector<uint>          &actuated_joints)
{
    std::vector<std::string> names;
    names.reserve(impl->model.joints.size());

    uint pin_joint_idx = 0;
    uint act_joint_idx = 0;
    for (pinocchio::JointIndex jid = 1; jid < impl->model.joints.size(); ++jid)
    {
        const auto &j = impl->model.joints[jid];

        if (j.nv() == 0) continue;

        std::string shortname = j.shortname();

        // Skip floating-base joints
        if (shortname == "JointModelFreeFlyer" ||
            shortname == "JointModelPlanar" ||
            shortname == "JointModelTranslation")
        {
            continue;
        }

        if (actuated_joints.empty() ||
            actuated_joints[act_joint_idx] == pin_joint_idx)
        {
            names.push_back(impl->model.names[jid]);
            if (++act_joint_idx == actuated_joints.size())
            {
                break;
            }
            ++pin_joint_idx;
            continue;
        }

        ++pin_joint_idx;
    }

    return names;
}

uint getTotalActuatedJoints(const std::shared_ptr<ModelImpl> &impl)
{
    uint total = 0;

    for (pinocchio::JointIndex jid = 1; jid < impl->model.joints.size(); ++jid)
    {
        const auto &j = impl->model.joints[jid];

        if (j.nv() == 0) continue;

        // Skip floating-base joints
        if (j.shortname() == "JointModelFreeFlyer" ||
            j.shortname() == "JointModelPlanar" ||
            j.shortname() == "JointModelTranslation")
        {
            continue;
        }

        ++total;
    }

    return total;
}

Eigen::VectorXd neutralPose(const std::shared_ptr<ModelImpl> &impl)
{
    return pinocchio::neutral(impl->model);
}

void dIntegrate(const std::shared_ptr<ModelImpl> &impl,
                const Eigen::VectorXd &q, const Eigen::VectorXd &v,
                Eigen::MatrixXd &J, const uint arg_pos)
{
    pinocchio::dIntegrate(impl->model,
                          q,
                          v,
                          J,
                          static_cast<pinocchio::ArgumentPosition>(arg_pos));
}

void dDifference(const std::shared_ptr<ModelImpl> &impl,
                 const Eigen::VectorXd &q0, const Eigen::VectorXd &q1,
                 Eigen::MatrixXd &J, const uint arg_pos)
{
    pinocchio::dDifference(impl->model,
                           q0,
                           q1,
                           J,
                           static_cast<pinocchio::ArgumentPosition>(arg_pos));
}

void updateFramePlacements(const std::shared_ptr<ModelImpl> &impl)
{
    pinocchio::updateFramePlacements(impl->model, impl->data);
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> forwardKinematics(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const std::string &frame)
{
    pinocchio::forwardKinematics(impl->model, impl->data, q);
    int frame_id = impl->model.getFrameId(frame);
    pinocchio::updateFramePlacement(impl->model, impl->data, frame_id);

    return { impl->data.oMf[frame_id].translation(),
             impl->data.oMf[frame_id].rotation() };
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> forwardKinematics(
  const std::shared_ptr<ModelImpl> &impl, const Eigen::VectorXd &q,
  const Eigen::VectorXd &v, const std::string &frame)
{
    pinocchio::forwardKinematics(impl->model, impl->data, q, v);
    int frame_id = impl->model.getFrameId(frame);
    pinocchio::updateFramePlacement(impl->model, impl->data, frame_id);

    return { impl->data.oMf[frame_id].translation(),
             impl->data.oMf[frame_id].rotation() };
}

Eigen::Matrix4d forwardKinematicsSE3(const std::shared_ptr<ModelImpl> &impl,
                                     const Eigen::VectorXd            &q,
                                     const std::string                &frame)
{
    pinocchio::forwardKinematics(impl->model, impl->data, q);
    int frame_id = impl->model.getFrameId(frame);
    pinocchio::updateFramePlacement(impl->model, impl->data, frame_id);

    return impl->data.oMf[frame_id].toHomogeneousMatrix();
}

Eigen::Vector<double, 6> frameVelocity(const std::shared_ptr<ModelImpl> &impl,
                                       const std::string &frame, bool local)
{
    return pinocchio::getFrameVelocity(
             impl->model,
             impl->data,
             impl->model.getFrameId(frame),
             local ? pinocchio::ReferenceFrame::LOCAL
                   : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
      .toVector();
}

void frameVelocityDerivatives(const std::shared_ptr<ModelImpl> &impl,
                              const Eigen::VectorXd            &q,
                              const Eigen::VectorXd &v, const std::string frame,
                              Eigen::MatrixXd &dVdq, Eigen::MatrixXd &dVdv,
                              bool local)
{
    pinocchio::computeForwardKinematicsDerivatives(
      impl->model,
      impl->data,
      q,
      v,
      Eigen::VectorXd::Zero(v.size()));

    Eigen::MatrixXd tmp1 = Eigen::MatrixXd::Zero(6, v.size());
    Eigen::MatrixXd tmp2 = Eigen::MatrixXd::Zero(6, v.size());
    Eigen::MatrixXd tmp3 = Eigen::MatrixXd::Zero(6, v.size());

    dVdq.setZero();
    dVdv.setZero();

    pinocchio::getFrameAccelerationDerivatives(
      impl->model,
      impl->data,
      impl->model.getFrameId(frame),
      local ? pinocchio::ReferenceFrame::LOCAL
            : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      dVdq,
      dVdv,
      tmp1,
      tmp2,
      tmp3);
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

Eigen::Vector3d centerOfMass(const std::shared_ptr<ModelImpl> &impl,
                             const Eigen::VectorXd            &q)
{
    return pinocchio::centerOfMass(impl->model, impl->data, q, false);
}

Eigen::MatrixXd jacobianCenterOfMass(const std::shared_ptr<ModelImpl> &impl,
                                     const Eigen::VectorXd            &q)
{
    return pinocchio::jacobianCenterOfMass(impl->model, impl->data, q, false);
}

Eigen::VectorXd randomConfiguration(const std::shared_ptr<ModelImpl> &impl)
{
    Eigen::VectorXd rc(impl->model.nq);
    pinocchio::randomConfiguration(impl->model,
                                   impl->model.lowerPositionLimit,
                                   impl->model.upperPositionLimit,
                                   rc);

    return rc;
}
}
}