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
#include <pinocchio/algorithm/regressor.hpp>

#include <boost/variant.hpp>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

namespace ssqp
{
namespace pinocchio_wrapper
{
// Results of kinematics passes at one configuration and velocity, so that
// queries at the same point share a single pass
struct KinematicsCache
{
    explicit KinematicsCache(const pinocchio::Model &model) : data(model) {}

    pinocchio::Data data;

    // Point the flags below hold results for
    VecX q, v;

    // Results for q
    bool placements = false;
    bool jacobians  = false;

    // Results for q and v
    bool velocities       = false;
    bool world_velocities = false;

    // Joint forces in Pinocchio's type, kept so that passes with forces don't
    // allocate them each time
    pinocchio::container::aligned_vector<pinocchio::Force> forces;
};

struct ModelImpl
{
    pinocchio::Model model;
    // Data must be constructed after Model. Constructing a temporary Data and
    // move-assigning it into Pinocchio's default Data reads uninitialized
    // Eigen solver state under sanitizers.
    // Used by the center of mass and regressor algorithms
    std::optional<pinocchio::Data> data;

    // Kinematics and dynamics caches, the default ones first and then one of
    // each per stage
    std::vector<std::unique_ptr<KinematicsCache>> kinematics_caches;
    std::vector<std::unique_ptr<DynamicsCache>>   dynamics_caches;
    // Index of the caches queries use
    uint active_cache = 0;
};

bool sameBits(ConstVecXRef a, ConstVecXRef b)
{
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

namespace
{
// Points the cache at configuration q, dropping results for any other one
void setConfiguration(KinematicsCache &cache, ConstVecXRef q)
{
    if (sameBits(cache.q, q))
    {
        return;
    }

    cache.q                = q;
    cache.placements       = false;
    cache.jacobians        = false;
    cache.velocities       = false;
    cache.world_velocities = false;
}

// Points the cache at velocity v, dropping results for any other one
void setVelocity(KinematicsCache &cache, ConstVecXRef v)
{
    if (sameBits(cache.v, v))
    {
        return;
    }

    cache.v                = v;
    cache.velocities       = false;
    cache.world_velocities = false;
}

// The functions below run a kinematics pass on the active cache only if its
// results for the given point are missing
// A pass also marks the results it computes along the way, which match those
// of the pass computing them alone up to roundoff

KinematicsCache &ensurePlacements(const std::shared_ptr<ModelImpl> &impl,
                                  ConstVecXRef                      q)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];
    setConfiguration(cache, q);

    if (!cache.placements)
    {
        pinocchio::forwardKinematics(impl->model, cache.data, q);
        cache.placements = true;
    }

    return cache;
}

KinematicsCache &ensureJacobians(const std::shared_ptr<ModelImpl> &impl,
                                 ConstVecXRef                      q)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];
    setConfiguration(cache, q);

    if (!cache.jacobians)
    {
        pinocchio::computeJointJacobians(impl->model, cache.data, q);
        cache.placements = true;
        cache.jacobians  = true;
    }

    return cache;
}

KinematicsCache &ensureVelocities(const std::shared_ptr<ModelImpl> &impl,
                                  ConstVecXRef q, ConstVecXRef v)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];
    setConfiguration(cache, q);
    setVelocity(cache, v);

    if (!cache.velocities)
    {
        pinocchio::forwardKinematics(impl->model, cache.data, q, v);
        cache.placements = true;
        cache.velocities = true;
    }

    return cache;
}

// Velocity derivatives only need the placements, the Jacobians and the joint
// velocities in the world frame, which follow from the local ones
KinematicsCache &ensureVelocityDerivatives(
  const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q, ConstVecXRef v)
{
    ensureJacobians(impl, q);
    KinematicsCache &cache = ensureVelocities(impl, q, v);

    if (!cache.world_velocities)
    {
        for (pinocchio::JointIndex i = 1;
             i < (pinocchio::JointIndex)impl->model.njoints;
             ++i)
        {
            cache.data.ov[i] = cache.data.oMi[i].act(cache.data.v[i]);
        }
        cache.world_velocities = true;
    }

    return cache;
}
}

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

Vec3 logSO3(const Mat3 &mat)
{
    return pinocchio::log3(mat);
}

Mat3 JlogSO3(const Mat3 &mat)
{
    Mat3 Jlog;
    pinocchio::Jlog3(mat, Jlog);
    return Jlog;
}

Vec6 logSE3(const Mat4 &mat)
{
    return pinocchio::log6(mat);
}

Mat6 JlogSE3(const Mat4 &mat)
{
    Mat6 Jlog;
    Jlog.setZero();
    pinocchio::Jlog6(pinocchio::SE3(mat), Jlog);
    return Jlog;
}

Mat6 SE3ToActionMatrixInverse(const Mat4 &mat)
{
    return pinocchio::SE3(mat).toActionMatrixInverse();
}

std::shared_ptr<ModelImpl> makeImpl()
{
    return std::make_shared<ModelImpl>();
}

void makeStageCaches(const std::shared_ptr<ModelImpl> &impl, const uint stages)
{
    impl->kinematics_caches.clear();
    impl->dynamics_caches.clear();

    for (uint i = 0; i <= stages; ++i)
    {
        impl->kinematics_caches.push_back(
          std::make_unique<KinematicsCache>(impl->model));
        impl->dynamics_caches.push_back(std::make_unique<DynamicsCache>());
    }

    impl->active_cache = 0;
}

uint activeCacheIndex(const std::shared_ptr<ModelImpl> &impl)
{
    return impl->active_cache;
}

void setActiveCacheIndex(const std::shared_ptr<ModelImpl> &impl,
                         const uint                        index)
{
    impl->active_cache = index;
}

DynamicsCache &activeDynamicsCache(const std::shared_ptr<ModelImpl> &impl)
{
    return *impl->dynamics_caches[impl->active_cache];
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
    impl->data.emplace(impl->model);
    makeStageCaches(impl, 0);
}

void buildModelFromUrdfXML(const std::string         &urdf,
                           std::shared_ptr<ModelImpl> impl, bool free_flyer)
{
    if (free_flyer)
    {
        pinocchio::urdf::buildModelFromXML(urdf,
                                           pinocchio::JointModelFreeFlyer(),
                                           impl->model);
    }
    else
    {
        pinocchio::urdf::buildModelFromXML(urdf, impl->model);
    }
    impl->data.emplace(impl->model);
    makeStageCaches(impl, 0);
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
    impl->data.emplace(impl->model);
    makeStageCaches(impl, 0);
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

std::pair<VecX, VecX> modelBounds(
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
    VecX lb(nbounds), ub(nbounds);

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

std::pair<VecX, VecX> modelEffortLimits(const std::shared_ptr<ModelImpl> &impl)
{
    return { -impl->model.effortLimit, impl->model.effortLimit };
}

VecX integrate(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
               ConstVecXRef v)
{
    VecX out(impl->model.nq);
    integrate(impl, q, v, out);
    return out;
}

void integrate(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
               ConstVecXRef v, VecXRef out)
{
    pinocchio::integrate(impl->model, q, v, out);
}

VecX difference(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q0,
                ConstVecXRef q1)
{
    VecX out(impl->model.nv);
    difference(impl, q0, q1, out);
    return out;
}

void difference(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q0,
                ConstVecXRef q1, VecXRef out)
{
    pinocchio::difference(impl->model, q0, q1, out);
}

void getFramejMfMatrix(const std::shared_ptr<ModelImpl> &impl,
                       const std::string &frame, uint &joint_id, Mat3 &rotation,
                       Vec3 &translation)
{
    pinocchio::FrameIndex frame_id = impl->model.getFrameId(frame);
    pinocchio::SE3        jMf      = impl->model.frames[frame_id].placement;
    joint_id                       = impl->model.frames[frame_id].parentJoint;
    rotation                       = jMf.rotation();
    translation                    = jMf.translation();
}

Vec6 framejMfAct(const std::shared_ptr<ModelImpl> &impl,
                 const std::string &frame, const Vec3 &force, uint &joint_id)
{
    return framejMfAct(impl, impl->model.getFrameId(frame), force, joint_id);
}

Vec6 framejMfAct(const std::shared_ptr<ModelImpl> &impl, const uint frame_id,
                 const Vec3 &force, uint &joint_id)
{
    pinocchio::Force f = pinocchio::Force::Zero();
    f.linear()         = force;

    pinocchio::Force res = impl->model.frames[frame_id].placement.act(f);
    joint_id             = impl->model.frames[frame_id].parentJoint;

    return res.toVector();
}

Vec6 framejMfActWrench(const std::shared_ptr<ModelImpl> &impl,
                       const uint frame_id, const Vec6 &wrench, uint &joint_id)
{
    pinocchio::Force f = pinocchio::Force::Zero();
    f.linear()         = wrench.head<3>();
    f.angular()        = wrench.tail<3>();

    pinocchio::Force res = impl->model.frames[frame_id].placement.act(f);
    joint_id             = impl->model.frames[frame_id].parentJoint;

    return res.toVector();
}

void crba(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q, MatX &M,
          bool local)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];
    setConfiguration(cache, q);

    pinocchio::crba(impl->model,
                    cache.data,
                    q,
                    local ? pinocchio::Convention::LOCAL
                          : pinocchio::Convention::WORLD);

    // The WORLD convention also leaves the placements and Jacobians
    if (!local)
    {
        cache.placements = true;
        cache.jacobians  = true;
    }

    M = cache.data.M.selfadjointView<Eigen::Upper>();
}

VecX aba(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q, ConstVecXRef v,
         ConstVecXRef tau, const std::vector<Vec6> &fext)
{
    VecX ddq(impl->model.nv);
    aba(impl, q, v, tau, fext, ddq);
    return ddq;
}

VecX aba(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q, ConstVecXRef v,
         ConstVecXRef tau)
{
    VecX ddq(impl->model.nv);
    aba(impl, q, v, tau, ddq);
    return ddq;
}

void aba(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q, ConstVecXRef v,
         ConstVecXRef tau, const std::vector<Vec6> &fext, VecXRef ddq)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];

    auto &forces = cache.forces;
    forces.resize(fext.size());
    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear()  = fext[i].head(3);
        forces[i].angular() = fext[i].tail(3);
    }

    setConfiguration(cache, q);
    setVelocity(cache, v);

    pinocchio::aba(impl->model, cache.data, q, v, tau, forces);

    // The pass leaves the joint velocities, which need the placements too
    cache.velocities = cache.placements;

    ddq = cache.data.ddq;
}

void aba(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q, ConstVecXRef v,
         ConstVecXRef tau, VecXRef ddq)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];
    setConfiguration(cache, q);
    setVelocity(cache, v);

    pinocchio::aba(impl->model, cache.data, q, v, tau);

    // The pass leaves the joint velocities, which need the placements too
    cache.velocities = cache.placements;

    ddq = cache.data.ddq;
}

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           ConstVecXRef q, ConstVecXRef v, ConstVecXRef tau,
                           MatX &aba_partial_dq, MatX &aba_partial_dv,
                           MatX &Minv)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];
    setConfiguration(cache, q);
    setVelocity(cache, v);

    pinocchio::computeABADerivatives(impl->model, cache.data, q, v, tau);

    // The pass leaves the placements, Jacobians and world velocities
    cache.placements       = true;
    cache.jacobians        = true;
    cache.world_velocities = true;

    aba_partial_dq = cache.data.ddq_dq;
    aba_partial_dv = cache.data.ddq_dv;
    Minv           = cache.data.Minv;
}

void computeABADerivatives(const std::shared_ptr<ModelImpl> &impl,
                           ConstVecXRef q, ConstVecXRef v, ConstVecXRef tau,
                           const std::vector<Vec6> &fext, MatX &aba_partial_dq,
                           MatX &aba_partial_dv, MatX &Minv)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];

    auto &forces = cache.forces;
    forces.resize(fext.size());
    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear()  = fext[i].head(3);
        forces[i].angular() = fext[i].tail(3);
    }

    setConfiguration(cache, q);
    setVelocity(cache, v);

    pinocchio::computeABADerivatives(impl->model,
                                     cache.data,
                                     q,
                                     v,
                                     tau,
                                     forces);

    // The pass leaves the placements, Jacobians and world velocities
    cache.placements       = true;
    cache.jacobians        = true;
    cache.world_velocities = true;

    aba_partial_dq = cache.data.ddq_dq;
    aba_partial_dv = cache.data.ddq_dv;
    Minv           = cache.data.Minv;
}

VecX rnea(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
          ConstVecXRef v, ConstVecXRef a)
{
    VecX tau(impl->model.nv);
    rnea(impl, q, v, a, tau);
    return tau;
}

VecX rnea(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
          ConstVecXRef v, ConstVecXRef a, const std::vector<Vec6> &fext)
{
    VecX tau(impl->model.nv);
    rnea(impl, q, v, a, fext, tau);
    return tau;
}

void rnea(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
          ConstVecXRef v, ConstVecXRef a, VecXRef tau)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];
    setConfiguration(cache, q);
    setVelocity(cache, v);

    pinocchio::rnea(impl->model, cache.data, q, v, a);

    // The pass leaves the joint velocities, which need the placements too
    cache.velocities = cache.placements;

    tau = cache.data.tau;
}

void rnea(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
          ConstVecXRef v, ConstVecXRef a, const std::vector<Vec6> &fext,
          VecXRef tau)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];

    auto &forces = cache.forces;
    forces.resize(fext.size());
    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear()  = fext[i].head<3>();
        forces[i].angular() = fext[i].tail<3>();
    }

    setConfiguration(cache, q);
    setVelocity(cache, v);

    pinocchio::rnea(impl->model, cache.data, q, v, a, forces);

    // The pass leaves the joint velocities, which need the placements too
    cache.velocities = cache.placements;

    tau = cache.data.tau;
}

void computeRNEADerivatives(const std::shared_ptr<ModelImpl> &impl,
                            ConstVecXRef q, ConstVecXRef v, ConstVecXRef a,
                            MatX &rnea_partial_dq, MatX &rnea_partial_dv,
                            MatX &rnea_partial_da)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];
    setConfiguration(cache, q);
    setVelocity(cache, v);

    pinocchio::computeRNEADerivatives(impl->model, cache.data, q, v, a);

    // The pass leaves every kinematics result
    cache.placements       = true;
    cache.jacobians        = true;
    cache.velocities       = true;
    cache.world_velocities = true;

    rnea_partial_dq = cache.data.dtau_dq;
    rnea_partial_dv = cache.data.dtau_dv;
    rnea_partial_da = cache.data.M;

    rnea_partial_da = rnea_partial_da.selfadjointView<Eigen::Upper>();
}

void computeRNEADerivatives(const std::shared_ptr<ModelImpl> &impl,
                            ConstVecXRef q, ConstVecXRef v, ConstVecXRef a,
                            const std::vector<Vec6> &fext,
                            MatX &rnea_partial_dq, MatX &rnea_partial_dv,
                            MatX &rnea_partial_da)
{
    KinematicsCache &cache = *impl->kinematics_caches[impl->active_cache];

    auto &forces = cache.forces;
    forces.resize(fext.size());
    for (uint i = 0; i < fext.size(); ++i)
    {
        forces[i].linear()  = fext[i].head<3>();
        forces[i].angular() = fext[i].tail<3>();
    }

    setConfiguration(cache, q);
    setVelocity(cache, v);

    pinocchio::computeRNEADerivatives(impl->model, cache.data, q, v, a, forces);

    // The pass leaves every kinematics result
    cache.placements       = true;
    cache.jacobians        = true;
    cache.velocities       = true;
    cache.world_velocities = true;

    rnea_partial_dq = cache.data.dtau_dq;
    rnea_partial_dv = cache.data.dtau_dv;
    rnea_partial_da = cache.data.M;

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

uint getFrameId(const std::shared_ptr<ModelImpl> &impl,
                const std::string                &frame)
{
    return impl->model.getFrameId(frame);
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

VecX neutralPose(const std::shared_ptr<ModelImpl> &impl)
{
    return pinocchio::neutral(impl->model);
}

void dIntegrate(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
                ConstVecXRef v, MatX &J, const uint arg_pos)
{
    pinocchio::dIntegrate(impl->model,
                          q,
                          v,
                          J,
                          static_cast<pinocchio::ArgumentPosition>(arg_pos));
}

void dDifference(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q0,
                 ConstVecXRef q1, MatX &J, const uint arg_pos)
{
    pinocchio::dDifference(impl->model,
                           q0,
                           q1,
                           J,
                           static_cast<pinocchio::ArgumentPosition>(arg_pos));
}

std::pair<Vec3, Mat3> forwardKinematics(const std::shared_ptr<ModelImpl> &impl,
                                        ConstVecXRef                      q,
                                        const std::string                &frame)
{
    return forwardKinematics(impl, q, impl->model.getFrameId(frame));
}

std::pair<Vec3, Mat3> forwardKinematics(const std::shared_ptr<ModelImpl> &impl,
                                        ConstVecXRef q, const uint frame_id)
{
    pinocchio::Data &data = ensurePlacements(impl, q).data;
    pinocchio::updateFramePlacement(impl->model, data, frame_id);

    return { data.oMf[frame_id].translation(), data.oMf[frame_id].rotation() };
}

std::pair<Vec3, Mat3> forwardKinematics(const std::shared_ptr<ModelImpl> &impl,
                                        ConstVecXRef q, ConstVecXRef v,
                                        const std::string &frame)
{
    return forwardKinematics(impl, q, v, impl->model.getFrameId(frame));
}

std::pair<Vec3, Mat3> forwardKinematics(const std::shared_ptr<ModelImpl> &impl,
                                        ConstVecXRef q, ConstVecXRef v,
                                        const uint frame_id)
{
    pinocchio::Data &data = ensureVelocities(impl, q, v).data;
    pinocchio::updateFramePlacement(impl->model, data, frame_id);

    return { data.oMf[frame_id].translation(), data.oMf[frame_id].rotation() };
}

Mat4 forwardKinematicsSE3(const std::shared_ptr<ModelImpl> &impl,
                          ConstVecXRef q, const std::string &frame)
{
    return forwardKinematicsSE3(impl, q, impl->model.getFrameId(frame));
}

Mat4 forwardKinematicsSE3(const std::shared_ptr<ModelImpl> &impl,
                          ConstVecXRef q, const uint frame_id)
{
    pinocchio::Data &data = ensurePlacements(impl, q).data;
    pinocchio::updateFramePlacement(impl->model, data, frame_id);

    return data.oMf[frame_id].toHomogeneousMatrix();
}

Vec6 frameVelocity(const std::shared_ptr<ModelImpl> &impl,
                   const std::string &frame, bool local)
{
    return frameVelocity(impl, impl->model.getFrameId(frame), local);
}

Vec6 frameVelocity(const std::shared_ptr<ModelImpl> &impl, const uint frame_id,
                   bool local)
{
    return pinocchio::getFrameVelocity(
             impl->model,
             impl->kinematics_caches[impl->active_cache]->data,
             frame_id,
             local ? pinocchio::ReferenceFrame::LOCAL
                   : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
      .toVector();
}

void frameVelocityDerivatives(const std::shared_ptr<ModelImpl> &impl,
                              ConstVecXRef q, ConstVecXRef v,
                              const std::string frame, MatXRef dVdq,
                              MatXRef dVdv, bool local)
{
    frameVelocityDerivatives(impl,
                             q,
                             v,
                             impl->model.getFrameId(frame),
                             dVdq,
                             dVdv,
                             local);
}

void frameVelocityDerivatives(const std::shared_ptr<ModelImpl> &impl,
                              ConstVecXRef q, ConstVecXRef v,
                              const uint frame_id, MatXRef dVdq, MatXRef dVdv,
                              bool local)
{
    pinocchio::Data &data = ensureVelocityDerivatives(impl, q, v).data;

    set_zero(dVdq);
    set_zero(dVdv);

    pinocchio::getFrameVelocityDerivatives(
      impl->model,
      data,
      frame_id,
      local ? pinocchio::ReferenceFrame::LOCAL
            : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      dVdq,
      dVdv);
}

void frameJacobian(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
                   const std::string &frame, MatXRef J, bool local)
{
    frameJacobian(impl, q, impl->model.getFrameId(frame), J, local);
}

void frameJacobian(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q,
                   const uint frame_id, MatXRef J, bool local)
{
    pinocchio::getFrameJacobian(
      impl->model,
      ensureJacobians(impl, q).data,
      frame_id,
      local ? pinocchio::ReferenceFrame::LOCAL
            : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J);
}

void computeJointJacobians(const std::shared_ptr<ModelImpl> &impl,
                           ConstVecXRef                      q)
{
    ensureJacobians(impl, q);
}

void frameJacobian(const std::shared_ptr<ModelImpl> &impl,
                   const std::string &frame, MatXRef J, bool local)
{
    frameJacobian(impl, impl->model.getFrameId(frame), J, local);
}

void frameJacobian(const std::shared_ptr<ModelImpl> &impl, const uint frame_id,
                   MatXRef J, bool local)
{
    pinocchio::getFrameJacobian(
      impl->model,
      impl->kinematics_caches[impl->active_cache]->data,
      frame_id,
      local ? pinocchio::ReferenceFrame::LOCAL
            : pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J);
}

Vec3 centerOfMass(const std::shared_ptr<ModelImpl> &impl, ConstVecXRef q)
{
    return pinocchio::centerOfMass(impl->model, *impl->data, q, false);
}

MatX jacobianCenterOfMass(const std::shared_ptr<ModelImpl> &impl,
                          ConstVecXRef                      q)
{
    MatX J(3, impl->model.nv);
    jacobianCenterOfMass(impl, q, J);
    return J;
}

void jacobianCenterOfMass(const std::shared_ptr<ModelImpl> &impl,
                          ConstVecXRef q, MatXRef J)
{
    J = pinocchio::jacobianCenterOfMass(impl->model, *impl->data, q, false);
}

VecX randomConfiguration(const std::shared_ptr<ModelImpl> &impl)
{
    VecX rc(impl->model.nq);
    pinocchio::randomConfiguration(impl->model,
                                   impl->model.lowerPositionLimit,
                                   impl->model.upperPositionLimit,
                                   rc);

    return rc;
}

VecX getAllDynamicParameters(const std::shared_ptr<ModelImpl> &impl)
{
    VecX parameters(10 * (impl->model.njoints - 1));

    for (uint i = 1; i < impl->model.njoints; ++i)
    {
        parameters.segment<10>(10 * (i - 1)) =
          impl->model.inertias[i].toDynamicParameters();
    }

    return parameters;
}

void setAllDynamicParameters(std::shared_ptr<ModelImpl> &impl,
                             const VecX                 &parameters)
{
    for (uint i = 1; i < impl->model.njoints; ++i)
    {
        const VecX parameters_i = parameters.segment<10>(10 * (i - 1));

        const double m   = parameters_i[0];
        const Vec3   com = parameters_i.segment<3>(1) / m;
        Mat3         I;

        I << parameters_i[4], parameters_i[5], parameters_i[6], parameters_i[5],
          parameters_i[7], parameters_i[8], parameters_i[6], parameters_i[8],
          parameters_i[9];

        impl->model.inertias[i] = pinocchio::Inertia(m, com, I);
    }

    // Dynamics results depend on the inertias, which the caches' keys don't
    // cover
    for (const std::unique_ptr<DynamicsCache> &cache : impl->dynamics_caches)
    {
        cache->clear();
    }
}

MatX computeJointTorqueRegressor(const std::shared_ptr<ModelImpl> &impl,
                                 ConstVecXRef q, ConstVecXRef v, ConstVecXRef a)
{
    return pinocchio::computeJointTorqueRegressor(impl->model,
                                                  *impl->data,
                                                  q,
                                                  v,
                                                  a);
}
}
}
