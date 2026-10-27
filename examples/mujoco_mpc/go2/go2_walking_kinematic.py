import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints
import argparse
import numpy as np
from utils_go2_mujoco import (
    ssqp_to_mujoco_state_kin,
    ssqp_to_mujoco_vel_kin,
    ssqp_to_mujoco_ctrl_kin,
    mujoco_to_ssqp_state_kin,
    clip_joints_to_limits,
)

import models.quadruped_hlplanner.planner as go2_planner

go2_planner.config.ContactScheduling.T = 25
go2_planner.config.ContactScheduling.hip_to_foot_offsets = {
    "FL": np.array((0.3, 0.14, -0.38797938)),
    "FR": np.array((0.3, -0.14, -0.38797938)),
    "RL": np.array((-0.15, 0.14, -0.38797938)),
    "RR": np.array((-0.15, -0.14, -0.38797938)),
}
go2_planner.config.ContactScheduling.calculate()

go2_planner.config.ReferenceGeneration.h_sw = 0.1

import multiprocessing
from time import sleep

from go2_controllers.joystick import Controller


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 kinematic walking MPC in MuJoCo using SSQP"
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm", "piqp"],
        default="admm",
        help="QP solver used at each SSQP iteration",
    )

    return parser.parse_args()


args = parse_args()

class KinematicJointSpacePD:
    def __init__(
        self,
        mpc_controller,
        init_state,
        Kp=1.0,
        Kd=0.01,
        update_func=lambda _1, _2, _3: None,
        max_vel_front=1.0,
        max_vel_back=1.0,
        max_vel_side=1.0,
        max_vel_yaw=1.0,
    ):
        self.mpc_controller = mpc_controller
        self.Kp = Kp
        self.Kd = Kd
        self.update_func = update_func

        self.last_mpc_sol = None
        self.mpc_traj_offset = 0

        self.state_q = multiprocessing.Queue()
        self.state_q.put((init_state, np.zeros(6)))

        self.mpc_q = multiprocessing.Queue()
        self.mpc_proc = multiprocessing.Process(
            target=self._run_mpc, daemon=True
        )

        self.max_vel_front = max_vel_front
        self.max_vel_back = max_vel_back
        self.max_vel_side = max_vel_side
        self.max_vel_yaw = max_vel_yaw

    def start_mpc(self):
        self.mpc_proc.start()

    def step(self, state, vel):
        self.state_q.put((state, vel))

        while self.last_mpc_sol is None and self.mpc_q.empty():
            sleep(0.1)

        if not self.mpc_q.empty():
            self.last_mpc_sol = self.mpc_q.get_nowait()
            self.mpc_traj_offset = 0

        control = self.Kp * (
            self.last_mpc_sol[0][self.mpc_traj_offset + 1][7:] - state[7:]
        ) + self.Kd * (
            self.last_mpc_sol[1][self.mpc_traj_offset + 1][6:] - vel[6:]
        )

        return control

    def _run_mpc(self):
        step = 0

        joy = Controller()

        while True:
            try:
                while self.state_q.empty():
                    sleep(0.001)

                while not self.state_q.empty():
                    state, vel = self.state_q.get()

                v_cmd = joy.read(
                    self.max_vel_front,
                    self.max_vel_back,
                    self.max_vel_side,
                    self.max_vel_yaw,
                )
                step += 1

                self.update_func(v_cmd, state, vel, step)

                next = [x.toarray() for x in self.mpc_controller.step(state)]

                self.mpc_q.put(next)

            except Exception as exc:
                print(exc)
                sleep(0.01)


ssqpy.setSilentMode()

v_cmd = np.array((0.0, 0.0, 0.0))

MPC_H = 10
dt = 0.02

model = ssqpy.model.Model(
    MPC_H,
    dt,
    mjcf_path="../../assets/mjcf/go2/go2.xml",
    free_flyer=True,
    solver_mode=ssqpy.model.SolverMode.Kinematics,
)

com_ref, xb_ref, p_ref = go2_planner.generate_references_MPC(
    v_cmd,
    go2_planner.config.INIT_CONFIG,
    np.zeros(3),
    0.0,
    MPC_H,
    dt,
    ssqp_model=model,
)

q_q = 1e-3
q_v = 1e-3
q_c = 1.5
q_b = 3.0
q_o = 1.0
q_sw = 1.5
q_sp = 10.0
l_sigma = 0.5

config_cost = costs.SquaredConfigurationErrorCost(
    model,
    go2_planner.config.INIT_JOINT_CONFIG,
    q_q,
    "Configuration Regularization Cost",
)

vel_cost = costs.SquaredControlCost(model, q_v, "Velocity Regularization Cost")

base_height_constraint = constraints.FrameTranslationBound(
    model,
    "base_link",
    np.array(
        (-np.inf, -np.inf, go2_planner.config.ReferenceGeneration.zb_min)
    ),
    np.array((np.inf, np.inf, go2_planner.config.ReferenceGeneration.zb_max)),
    "Base Height Constraint",
)

com_costs = []
pose_costs = []
feet_costs = {foot: [] for foot in go2_planner.config.FRAME_NAMES}
feet_constraints = {foot: [] for foot in go2_planner.config.FRAME_NAMES}
line_segment_costs = []

for k in range(MPC_H + 1):
    com_cost = costs.CenterOfMassTrackingCost(
        model, com_ref[k], q_c, f"CoM Cost {k}"
    )

    com_costs.append(com_cost)
    model.addCost(k, com_cost)

    pose_cost = costs.FrameSquaredPoseErrorCost(
        model,
        "base_link",
        xb_ref[k][:3],
        go2_planner.utils.quat_to_rotation_matrix(xb_ref[k][3:7]),
        q_b,
        q_o,
        f"Pose Cost {k}",
    )

    pose_costs.append(pose_cost)
    model.addCost(k, pose_cost)

    for foot in p_ref:
        foot_translation_cost = costs.FrameSquaredTranslationErrorCost(
            model,
            go2_planner.config.FRAME_NAMES[foot],
            p_ref[foot][k][0],
            q_sw,
            f"Foot {foot} Swing Cost {k}",
        )

        feet_costs[foot].append(foot_translation_cost)
        model.addCost(k, foot_translation_cost)

        foot_translation_bound = constraints.FrameTranslationBound(
            model,
            go2_planner.config.FRAME_NAMES[foot],
            p_ref[foot][k][0],
            p_ref[foot][k][0],
            f"Foot {foot} Stance Constraint {k}",
        )

        feet_constraints[foot].append(foot_translation_bound)
        # Stage 0 is the measured state, which the solver cannot move
        if k > 0:
            model.addConstraints(k, foot_translation_bound)

        if p_ref[foot][k][1]:  # swing
            foot_translation_bound.set_enabled(False)
        else:  # stance
            foot_translation_cost.set_enabled(False)

    line_segment_cost = go2_planner.LineSegmentCost(
        model,
        {foot: p_ref[foot][k] for foot in p_ref},
        q_sp,
        l_sigma,
        terminal=k == MPC_H,
    )

    line_segment_costs.append(line_segment_cost)
    model.addCost(k, line_segment_cost)

    model.addCost(k, config_cost)

    if k != MPC_H:
        model.addCost(k, vel_cost)

    if k > 0:
        model.addConstraints(k, base_height_constraint)

model.finalize(
    init_states=[go2_planner.config.INIT_CONFIG],
)

ssqp_params = ssqpy.solvers.ssqpParams()
ssqp_params.qp_solver_type = getattr(
    ssqpy.solvers.QPSolverType, args.qp_solver.upper()
)
ssqp_params.tolerance = 1e-1

admm_params = ssqpy.solvers.admmParams()
admm_params.abs_tolerance = 1e-2
admm_params.rel_tolerance = 1e-2
admm_params.warm_start = True
admm_params.init_rho = 10.0
admm_params.adaptive_rho_tolerance = 0.1

ssqp_params.admmParams = admm_params

mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=3, qp_iters=200)

np.set_printoptions(suppress=True)

# ssqpy.setLogLevel(ssqpy.LogLevel.Debug)


def update_func(v_cmd, config, vel, step):
    vel_3 = np.hstack((vel[:2], vel[5]))
    com_ref, xb_ref, p_ref = go2_planner.generate_references_MPC(
        v_cmd,
        config,
        vel_3,
        step,
        MPC_H,
        dt,
        ssqp_model=model,
    )

    for k in range(MPC_H + 1):
        com_costs[k].set_cref(com_ref[k])
        pose_costs[k].set_desired_translation(xb_ref[k][:3])
        pose_costs[k].set_desired_rotation(
            go2_planner.utils.quat_to_rotation_matrix(xb_ref[k][3:7])
        )

        stage_p_ref = {foot: p_ref[foot][k] for foot in p_ref}

        for foot in go2_planner.config.FRAME_NAMES:
            if stage_p_ref[foot][1]:  # swing
                feet_constraints[foot][k].set_enabled(False)
                feet_costs[foot][k].set_enabled(True)
                feet_costs[foot][k].set_desired_translation(
                    stage_p_ref[foot][0]
                )
            else:  # stance
                feet_costs[foot][k].set_enabled(False)
                feet_constraints[foot][k].set_enabled(True)
                feet_constraints[foot][k].set_lower_bound(stage_p_ref[foot][0])
                feet_constraints[foot][k].set_upper_bound(stage_p_ref[foot][0])

        line_segment_costs[k].update_reference(stage_p_ref)


Kp = 40.0
Kv = 2.0 * np.sqrt(Kp)

joint_space_pd_controller = KinematicJointSpacePD(
    mpc,
    go2_planner.config.INIT_CONFIG,
    Kp=Kp,
    Kd=Kv,
    update_func=update_func,
    max_vel_front=go2_planner.config.ReferenceGeneration.max_vel_front,
    max_vel_back=go2_planner.config.ReferenceGeneration.max_vel_back,
    max_vel_side=go2_planner.config.ReferenceGeneration.max_vel_side,
    max_vel_yaw=go2_planner.config.ReferenceGeneration.max_vel_yaw,
)

joint_space_pd_controller.start_mpc()

import mujoco
import mujoco.viewer
import glfw

mjmodel = mujoco.MjModel.from_xml_path("../../assets/mjcf/go2/scene.xml")
mjdata = mujoco.MjData(mjmodel)

mjdata.qpos = ssqp_to_mujoco_state_kin(go2_planner.config.INIT_CONFIG)

# Start the viewer maximized
glfw.init()
glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
    while viewer.is_running():
        pos = mujoco_to_ssqp_state_kin(mjdata.qpos)
        pos = clip_joints_to_limits(pos, mjmodel)
        vel = ssqp_to_mujoco_vel_kin(mjdata.qvel)

        mjdata.ctrl = ssqp_to_mujoco_ctrl_kin(
            joint_space_pd_controller.step(pos, vel)
        )

        mujoco.mj_step(mjmodel, mjdata)
        viewer.sync()

print()
