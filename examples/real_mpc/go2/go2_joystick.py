import os
import time
import numpy as np
from threading import Lock, RLock, Thread

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient

from threading import Event

import kinematic_walk_model
import dynamic_walk_model

from mujoco_mpc.go2.go2_controllers.joystick import Controller

LegID = {
    "FR_0": 0,
    "FR_1": 1,
    "FR_2": 2,
    "FL_0": 3,
    "FL_1": 4,
    "FL_2": 5,
    "RR_0": 6,
    "RR_1": 7,
    "RR_2": 8,
    "RL_0": 9,
    "RL_1": 10,
    "RL_2": 11,
}

# Performance cores of hybrid CPUs, as reported by the kernel. The solver
# thread runs on these, since solves on efficiency cores are several times
# slower than the walk tolerates
PERFORMANCE_CORES_PATH = "/sys/devices/cpu_core/cpus"


def performance_cores():
    # Returns the set of performance cores, or None if the CPU is not hybrid
    if not os.path.exists(PERFORMANCE_CORES_PATH):
        return None

    cores = set()
    with open(PERFORMANCE_CORES_PATH) as f:
        for part in f.read().strip().split(","):
            first, _, last = part.partition("-")
            cores.update(range(int(first), int(last or first) + 1))
    return cores


HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0


class CustomGo2Controller:
    JOINT_REORDER_MAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    JOINT_Kps = [
        30.0,
        40.0,
        40.0,
        30.0,
        40.0,
        40.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
    ]

    JOINT_Kds = [0.3 * np.sqrt(Kp) for Kp in JOINT_Kps]

    # Fallback, which moves to the stance position, then lies down and damps
    # the joints. It stays on until the robot stands up again
    FALLBACK_STAND_TIME = 0.5
    FALLBACK_DROP_TIME = 2.5
    DAMPING_KD = 2.0

    # Gamepad button that triggers the fallback, and once lying down stands
    # the robot up again
    FALLBACK_BUTTON = "BTN_SOUTH"

    # Gamepad buttons held during the dynamic walk to tilt the front of the
    # base up or down, tilt the base to the left or right, and turn the front
    # to the left or right
    PITCH_UP_BUTTON = "BTN_Z"
    PITCH_DOWN_BUTTON = "BTN_TR"
    ROLL_LEFT_BUTTON = "BTN_NORTH"
    ROLL_RIGHT_BUTTON = "BTN_EAST"
    YAW_LEFT_BUTTON = "BTN_WEST"
    YAW_RIGHT_BUTTON = "BTN_TL"

    # Dynamic walk limits that trigger the fallback. The joint tracking error
    # has to stay above its limit for a few commands in a row, and the tilt is
    # measured from the planned one
    MAX_PLAN_AGE = 0.1
    MAX_STATE_AGE = 0.05
    MAX_TRACKING_ERROR = 0.3
    TRACKING_ERROR_COMMANDS = 3
    MAX_TILT = 0.35

    def __init__(self):
        self.Kp = 40.0
        self.Kd = 1.6
        self.dt = 0.01

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.low_state_time = None
        self.current_base_pos = (
            kinematic_walk_model.go2_planner.config.INIT_BASE_POS
        )
        self.current_base_vel = np.zeros(3)

        self.state_lock = Lock()
        self.cmd_lock = RLock()

        # Gamepad for the walks and the fallback button
        self.joystick = Controller()

        # Position before starting stand move
        self._stand_ground_pos = [
            0.0,
            1.36,
            -2.65,
            0.0,
            1.36,
            -2.65,
            -0.2,
            1.66,
            -2.65,
            0.2,
            1.66,
            -2.65,
        ]

        # Stance position
        self._stance_pos = [
            0.0,
            0.67,
            -1.3,
            0.0,
            0.67,
            -1.3,
            0.0,
            0.67,
            -1.3,
            0.0,
            0.67,
            -1.3,
        ]

        # Position after dropping from stance
        self._drop_pos = [
            -0.35,
            1.36,
            -2.65,
            0.35,
            1.36,
            -2.65,
            -0.5,
            1.36,
            -2.65,
            0.5,
            1.36,
            -2.65,
        ]

        # Starting position
        self.start_pos = [0.0] * 12

        # Current task
        self.task = None

        # Time the fallback started, if it has
        self.fallback_start = None
        self.fallback_button_held = False

        self.stand_start = None

        def stand_task(self):
            self.first_run = True
            self.percent_1 = 0.0
            self.percent_2 = 0.0
            self.percent_3 = 0.0

        def drop_task(self):
            self.first_run = True
            self.percent_1 = 0.0
            self.percent_2 = 0.0

        def kinematic_walk_task(self):
            self.step = 0
            self.stance_feet = []

            self.model, self.mpc_controller, self.update_func = (
                kinematic_walk_model.get_controller(
                    self._getGeneralizedConfiguration(self.current_base_pos)
                )
            )

        def dynamic_walk_task(self):
            (
                self.model,
                self.mpc_controller,
                self.update_func,
                self.plan_commands,
                self.command_torques,
                self.filter_state,
            ) = dynamic_walk_model.get_controller(
                self._getGeneralizedConfiguration(self.current_base_pos)
            )

            # Latest plan with the time it started
            self.plan = None
            self.tracking_error_commands = 0

            # Solve in a separate thread, so that commands keep going out and
            # the state keeps updating while it solves
            self.solver_running = True
            self.solver_thread = Thread(
                target=self._runDynamicWalkSolver, daemon=True
            )
            self.solver_thread.start()

        self.task_map = {
            None: (lambda obj: None, lambda: None),
            "stand": (stand_task, self._execStandTask),
            "drop": (drop_task, self._execDropTask),
            "kinematic-walk": (
                kinematic_walk_task,
                self._execKinematicWalkTask,
            ),
            "dynamic-walk": (
                dynamic_walk_task,
                self._execDynamicWalkTask,
            ),
        }

        # Thread handling
        self.low_cmd_write_thread = None

        self.crc = CRC()

        # Inititalize `low_cmd`
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

        # Create publisher
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # Create subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._lowStateMessageHandler, 1)

        # self.sportmode_subscriber = ChannelSubscriber(
        #     "rt/sportmodestate", SportModeState_
        # )
        # self.sportmode_subscriber.Init(self._sportStateMessageHandler, 1)

        self.sc = SportClient()
        self.sc.SetTimeout(1.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(1.0)
        self.msc.Init()

        self.lieAndReleaseMode()

    def lieAndReleaseMode(self):
        _, result = self.msc.CheckMode()

        if result is None:
            return

        while result["name"]:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            _, result = self.msc.CheckMode()
            time.sleep(1)

    def start(self):
        self.low_cmd_write_thread = RecurrentThread(
            interval=self.dt,
            target=self._lowCmdWrite,
            name="writebasiccmd",  # TODO: dt=0.009 works
        )
        self.low_cmd_write_thread.Start()

    def join(self):
        Event().wait()

    def set_task(self, task):
        if task not in self.task_map.keys():
            raise ValueError("Invalid task")

        if self.fallback_start is not None:
            print(f"Ignoring the {task} task during the fallback")
            return

        # Stop the dynamic walk solver, if it is running
        self.solver_running = False
        if getattr(self, "solver_thread", None) is not None:
            self.solver_thread.join()
            self.solver_thread = None

        # Initialize task via task initializer function
        self.task_map[task][0](self)
        # Set task as the current task
        self.task = task

    # Private methods
    def _lowStateMessageHandler(self, msg: LowState_):
        filtering = (
            self.task == "dynamic-walk" and dynamic_walk_model.USE_FILTER
        )
        self.state_lock.acquire()
        self.low_state = msg
        self.low_state_time = time.time()
        if filtering:
            pos = self._getGeneralizedConfiguration(self.current_base_pos)
            vel = self._getGeneralizedVelocity(self.current_base_vel)
            acc = np.array(msg.imu_state.accelerometer)
        self.state_lock.release()

        # Update the state filter with every message
        if filtering:
            self.filter_state(pos, vel, acc, time.time())

    def _sportStateMessageHandler(self, msg: SportModeState_):
        self.state_lock.acquire()
        self.current_base_pos = np.hstack((np.zeros(2), msg.position[2]))
        self.current_base_vel = np.array(msg.velocity)
        self.state_lock.release()

    def _lowCmdWrite(self):
        if self.low_state is None:
            return

        # The dynamic walk's solver thread also writes commands
        with self.cmd_lock:
            # React to each new press of the fallback button
            held = self.FALLBACK_BUTTON in self.joystick.buttons
            if held and not self.fallback_button_held:
                if self.fallback_start is None:
                    if self.stand_start is None:
                        self._startFallback("the fallback button was pressed")
                elif (
                    time.time() - self.fallback_start
                    > self.FALLBACK_STAND_TIME + self.FALLBACK_DROP_TIME
                ):
                    print("Standing up after the fallback")
                    self.fallback_start = None
                    self.stand_start = time.time()
                    self.set_task("stand")
            self.fallback_button_held = held

            # Execute task method, unless the fallback has taken over
            if self.fallback_start is not None and self.stand_start is None:
                self._execFallbackTask()
            elif (
                self.stand_start is not None
                and time.time() - self.stand_start > 3.0
            ):
                self.stand_start = None
                self.set_task("dynamic-walk")
            else:
                self.task_map[self.task][1]()

            # Publish generated commands
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

    def _execStandTask(self):
        # On first run, copy current motor state
        if self.first_run:
            for i in range(12):
                self.start_pos[i] = self.low_state.motor_state[i].q
            self.first_run = False

        # On sub-move 1, interpolate between initial and stand-ground positions
        self.percent_1 += self.dt / 2.0
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (
                    1 - self.percent_1
                ) * self.start_pos[
                    i
                ] + self.percent_1 * self._stand_ground_pos[
                    i
                ]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # On sub-move 2, interpolate between stand-ground and stance positions
        if self.percent_2 < 1:
            self.percent_2 += self.dt / 2.0
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (
                    1 - self.percent_2
                ) * self._stand_ground_pos[
                    i
                ] + self.percent_2 * self._stance_pos[
                    i
                ]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # On sub-move 3, hold stance position
        elif self.percent_3 < 1:
            self.percent_3 += self.dt / 2.0
            self.percent_3 = min(self.percent_3, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self._stance_pos[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # On finish, revert to None task
        else:
            self.task = None

    def _execDropTask(self):
        # On first run, copy current motor state
        if self.first_run:
            for i in range(12):
                self.start_pos[i] = self.low_state.motor_state[i].q
            self.first_run = False

        # On sub-move 1, interpolate between initial and stance positions
        # self.percent_1 += self.dt / 2.5
        # self.percent_1 = min(self.percent_1, 1)
        # if self.percent_1 < 1:
        #     for i in range(12):
        #         self.low_cmd.motor_cmd[i].q = (
        #             1 - self.percent_1
        #         ) * self.start_pos[i] + self.percent_1 * self._stance_pos[i]
        #         self.low_cmd.motor_cmd[i].dq = 0
        #         self.low_cmd.motor_cmd[i].kp = self.Kp
        #         self.low_cmd.motor_cmd[i].kd = self.Kd
        #         self.low_cmd.motor_cmd[i].tau = 0

        # On sub-move 2, interpolate between stance and ground positions
        if self.percent_2 < 1:
            self.percent_2 += self.dt / 2.5
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (
                    1 - self.percent_2
                ) * self._stance_pos[i] + self.percent_2 * self._drop_pos[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # On finish, revert to None task
        else:
            self.task = None

    def _execKinematicWalkTask(self):
        # Get generalized position and velocity vectors
        self.state_lock.acquire()
        pos = self._getGeneralizedConfiguration(self.current_base_pos)
        vel = self._getGeneralizedVelocity(self.current_base_vel)
        self.state_lock.release()

        # Estimate the base height from the feet in stance, which are on the
        # ground, using forward kinematics with the base at the origin
        if self.stance_feet:
            local = pos.copy()
            local[:3] = 0.0
            height = -np.mean(
                [
                    self.model.forwardKinematics(
                        local,
                        kinematic_walk_model.go2_planner.config.FRAME_NAMES[
                            foot
                        ],
                    )[0][2]
                    for foot in self.stance_feet
                ]
            )
            # Offset height by some amount
            pos[2] = height + 0.05

        limits = kinematic_walk_model.go2_planner.config.ReferenceGeneration
        v_cmd = self.joystick.read(
            limits.max_vel_front,
            limits.max_vel_back,
            limits.max_vel_side,
            limits.max_vel_yaw,
        )
        self.step += 1

        p_ref = self.update_func(
            v_cmd,
            pos,
            vel,
            self.step,
        )

        self.stance_feet = []
        for foot in p_ref:
            if not p_ref[foot][0][1]:  # if stance
                self.stance_feet.append(foot)

        next = [x.toarray() for x in self.mpc_controller.step(pos)]

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = next[0][1][
                7 + self.JOINT_REORDER_MAP[i]
            ]
            self.low_cmd.motor_cmd[i].dq = next[1][1][
                6 + self.JOINT_REORDER_MAP[i]
            ]
            self.low_cmd.motor_cmd[i].kp = self.JOINT_Kps[i]
            self.low_cmd.motor_cmd[i].kd = self.JOINT_Kds[i]
            self.low_cmd.motor_cmd[i].tau = 0.0

        # time.sleep(0.01)

    def _runDynamicWalkSolver(self):
        cores = performance_cores()
        if cores is not None:
            os.sched_setaffinity(0, cores)

        next_solve = time.time()

        while self.solver_running:
            delay = next_solve - time.time()
            if delay > 0:
                time.sleep(delay)

            # Get generalized position and velocity vectors
            now = time.time()
            self.state_lock.acquire()
            pos = self._getGeneralizedConfiguration(self.current_base_pos)
            vel = self._getGeneralizedVelocity(self.current_base_vel)
            self.state_lock.release()

            v_cmd = self.joystick.read(
                dynamic_walk_model.MAX_VEL_FRONT,
                dynamic_walk_model.MAX_VEL_BACK,
                dynamic_walk_model.MAX_VEL_SIDE,
                dynamic_walk_model.MAX_VEL_YAW,
            )

            # Pitch, roll and yaw commands from pairs of opposite buttons
            buttons = self.joystick.buttons
            orientation = [
                (positive in buttons) - (negative in buttons)
                for positive, negative in [
                    (self.PITCH_UP_BUTTON, self.PITCH_DOWN_BUTTON),
                    (self.ROLL_RIGHT_BUTTON, self.ROLL_LEFT_BUTTON),
                    (self.YAW_LEFT_BUTTON, self.YAW_RIGHT_BUTTON),
                ]
            ]

            # Solve from the state predicted for when the plan starts
            start = now + dynamic_walk_model.PREDICT_TIME
            plan = self.plan

            try:
                state = self.update_func(
                    v_cmd,
                    self.joystick.dpad_x,
                    self.joystick.dpad_y,
                    orientation,
                    pos,
                    vel,
                )
                if plan is not None:
                    state = dynamic_walk_model.predict_state(
                        state, plan[1], now - plan[0], start - plan[0]
                    )
                x, u = self.mpc_controller.step(state)
                commands = self.plan_commands(x, u)
            except Exception as err:
                self._startFallback(f"the solver failed: {err}")
                return

            # Apply the plan as soon as it starts, rather than at the next
            # command
            delay = start - time.time()
            if delay > 0:
                time.sleep(delay)
            self.plan = (start, commands)
            self._lowCmdWrite()

            # Keep a fixed solve rate, but skip stages that are already late
            next_solve = max(
                next_solve + dynamic_walk_model.MPC_DT, time.time()
            )

    def _execDynamicWalkTask(self):
        # Keep the previous commands until the first plan is ready
        plan = self.plan
        if plan is None:
            return

        with self.state_lock:
            pos = self._getGeneralizedConfiguration(self.current_base_pos)
            vel = self._getGeneralizedVelocity(self.current_base_vel)
            state_time = self.low_state_time

        # Play the latest plan back by the time elapsed since it started
        now = time.time()
        elapsed = now - plan[0]
        q, dq = dynamic_walk_model.playback(plan[1], elapsed)
        tau = self.command_torques(plan[1], elapsed, pos, vel)

        if np.max(np.abs(q - pos[7:])) > self.MAX_TRACKING_ERROR:
            self.tracking_error_commands += 1
        else:
            self.tracking_error_commands = 0

        # Cosine of the angle between the measured and planned base tilts,
        # comparing the world z axis as seen from the base, which ignores yaw
        planned = dynamic_walk_model.plan_state(plan[1][0], elapsed)
        cos_tilt = (
            dynamic_walk_model.quat_to_rotation(pos[3:7])[2]
            @ dynamic_walk_model.quat_to_rotation(planned[3:7])[2]
        )

        reason = None
        if elapsed > self.MAX_PLAN_AGE:
            reason = "the solver stalled"
        elif now - state_time > self.MAX_STATE_AGE:
            reason = "the state stopped updating"
        elif not np.all(np.isfinite(np.hstack((q, dq, tau)))):
            reason = "the commands are not finite"
        elif self.tracking_error_commands >= self.TRACKING_ERROR_COMMANDS:
            reason = "the joints stopped following the plan"
        elif cos_tilt < np.cos(self.MAX_TILT):
            reason = "the base tilted too far from the plan"

        if reason is not None:
            self._startFallback(reason)
            self._execFallbackTask()
            return

        for i in range(12):
            j = self.JOINT_REORDER_MAP[i]
            self.low_cmd.motor_cmd[i].q = q[j]
            self.low_cmd.motor_cmd[i].dq = dq[j]
            self.low_cmd.motor_cmd[i].kp = dynamic_walk_model.JOINT_KP
            self.low_cmd.motor_cmd[i].kd = dynamic_walk_model.JOINT_KD
            self.low_cmd.motor_cmd[i].tau = tau[j]

    def _startFallback(self, reason):
        with self.cmd_lock:
            if self.fallback_start is not None:
                return

            print(f"Falling back since {reason}")
            self.solver_running = False
            for i in range(12):
                self.start_pos[i] = self.low_state.motor_state[i].q
            self.fallback_start = time.time()

    def _execFallbackTask(self):
        elapsed = time.time() - self.fallback_start

        # Move the joints to the stance position, then to the lying position
        if elapsed < self.FALLBACK_STAND_TIME + self.FALLBACK_DROP_TIME:
            if elapsed < self.FALLBACK_STAND_TIME:
                percent = elapsed / self.FALLBACK_STAND_TIME
                start, end = self.start_pos, self._stance_pos
            else:
                percent = (
                    elapsed - self.FALLBACK_STAND_TIME
                ) / self.FALLBACK_DROP_TIME
                start, end = self._stance_pos, self._drop_pos
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - percent) * start[
                    i
                ] + percent * end[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # Once lying down, damp the joints
        else:
            for i in range(12):
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = self.DAMPING_KD
                self.low_cmd.motor_cmd[i].tau = 0.0

    def _getGeneralizedConfiguration(self, base_pos):
        # Extract & reorder quaternion
        quat = np.array(self.low_state.imu_state.quaternion)
        w = quat[0]
        quat[:3] = quat[1:]
        quat[3] = w
        quat /= np.linalg.norm(quat)

        # Extract & reorder joint angles
        joints = []
        for i in range(12):
            joints.append(
                self.low_state.motor_state[self.JOINT_REORDER_MAP[i]].q
            )

        return np.hstack((base_pos, quat, joints))

    def _getGeneralizedVelocity(self, base_vel):
        # Extract angular velocity
        ang_vel = list(self.low_state.imu_state.gyroscope)

        joint_vel = []
        for i in range(12):
            joint_vel.append(
                self.low_state.motor_state[self.JOINT_REORDER_MAP[i]].dq
            )

        return np.hstack((base_vel, ang_vel, joint_vel))


if __name__ == "__main__":
    ChannelFactoryInitialize(0, "lo")

    custom = CustomGo2Controller()
    custom.start()

    custom.set_task("stand")
    time.sleep(3.0)

    custom.set_task("dynamic-walk")
    custom.join()
