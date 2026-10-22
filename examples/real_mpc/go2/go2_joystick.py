import gc
import os
import subprocess
import sys
import time
import argparse
import numpy as np
from threading import Event, Lock, RLock, Thread

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient

import kinematic_walk_model
import dynamic_walk_model as dwm
import elevation_map as em

from mujoco_mpc.go2.go2_controllers.joystick import Controller

ROBOT_INTERFACE = "enp5s0"
SIMULATION_INTERFACE = "lo"

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


def blend(start, end, share):
    # The joint positions the given share of the way from start to end
    return (1 - share) * np.asarray(start) + share * np.asarray(end)


PosStopF = 2.146e9
VelStopF = 16000.0


class CustomGo2Controller:
    JOINT_REORDER_MAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    # Joint gains of the kinematic walk
    JOINT_Kps = [30.0, 40.0, 40.0] * 2 + [30.0] * 6
    JOINT_Kds = [0.3 * np.sqrt(Kp) for Kp in JOINT_Kps]

    # Joint positions to start the stand move from, to stand at, and to drop
    # to from standing, by leg in the robot's order (FR, FL, RR and RL)
    STAND_GROUND_POS = np.ravel(
        [
            [0.0, 1.36, -2.65],
            [0.0, 1.36, -2.65],
            [-0.2, 1.66, -2.65],
            [0.2, 1.66, -2.65],
        ]
    )
    STANCE_POS = np.tile([0.0, 0.67, -1.3], 4)
    DROP_POS = np.ravel(
        [
            [-0.35, 1.36, -2.65],
            [0.35, 1.36, -2.65],
            [-0.5, 1.36, -2.65],
            [0.5, 1.36, -2.65],
        ]
    )

    # The robot measures neither its base position nor its linear velocity,
    # so the measured configurations take this base position, and the
    # measured velocities a base at rest
    BASE_POS = kinematic_walk_model.go2_planner.config.INIT_BASE_POS

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

    # Gamepad button that makes the dynamic walk jump in place while standing
    JUMP_BUTTON = "BTN_C"
    FORWARD_JUMP_BUTTON = "BTN_START"

    # Gamepad button that switches the dynamic walk to its next gait while
    # standing
    GAIT_BUTTON = "BTN_TL2"

    # Dynamic walk limits that trigger the fallback. The joint tracking error
    # has to stay above its limit for a few commands in a row, more of them for
    # swinging legs, which lag behind for a while whenever their foothold
    # moves, and the tilt is measured from the planned one
    MAX_PLAN_AGE = 0.1
    MAX_STATE_AGE = 0.05
    MAX_TRACKING_ERROR = 0.3
    TRACKING_ERROR_COMMANDS = 3
    SWING_TRACKING_ERROR_COMMANDS = 5
    MAX_TILT = 0.35

    # Foot sensor reading above which a foot counts as on the ground. The
    # sensors read about 12 in the air and 30 or more when loaded. The
    # simulator reports the force in newtons if its bridge fills it in, and
    # leaves it at zero otherwise, in which case it goes without foot contacts
    FOOT_CONTACT_FORCE = 20
    # Unitree foot sensor order (FR, FL, RR, RL) in the model's foot order
    FOOT_REORDER_MAP = [1, 0, 3, 2]

    # Seconds between printouts of the dynamic walk's MPC rate, if shown
    RATE_PERIOD = 2.0

    # LIDAR maps the dynamic walk reads after it starts before it uses them.
    # The map starts over with the walk, and its first maps come from only a
    # few sweeps, with the ground around the feet not yet seen
    MAP_SETTLE_MAPS = 10

    def __init__(
        self,
        interface,
        walk_task="dynamic-walk",
        map_view=False,
        map_save=None,
        use_map=True,
        show_rate=False,
    ):
        # Task to switch to once the robot is standing
        self.walk_task = walk_task
        # Whether the dynamic walk reads the LIDAR map at all, rather than
        # taking the ground to be flat
        self.use_map = use_map
        # Whether the dynamic walk prints its MPC rate and solve times
        self.show_rate = show_rate

        # Shared memory the dynamic walk's state estimate goes to the LIDAR
        # map process (lidar_map.py) through, and the command that runs it,
        # showing the map in a Rerun viewer or saving it to a .rrd file if set
        self.shared_poses = em.SharedPoses(em.POSES_NAME, create=True)
        mapper_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lidar_map.py"
        )
        self.mapper_command = [sys.executable, mapper_path, interface]
        if map_view:
            self.mapper_command.append("--view")
        if map_save:
            self.mapper_command += ["--save", map_save]
        # The map process runs when the walk uses the map or the map is shown
        # or saved
        self.run_mapper = use_map or map_view or bool(map_save)
        self.mapper = None
        # The map process's latest map, once it has made one
        self.shared_map = None

        self.Kp = 40.0
        self.Kd = 1.6
        self.dt = 0.01

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.low_state_time = None
        self.sim_foot_forces = False

        self.state_lock = Lock()
        self.cmd_lock = RLock()

        # Gamepad for the walks and the fallback button
        self.joystick = Controller()

        # Starting position
        self.start_pos = [0.0] * 12

        # Current task
        self.task = None

        # Time the fallback started, if it has
        self.fallback_start = None
        self.fallback_button_held = False

        self.stand_start = None

        # The dynamic walk's solver thread, while it runs
        self.solver_running = False
        self.solver_thread = None

        # Each task's initializer and the function giving its commands
        self.task_map = {
            None: (lambda: None, lambda: None),
            "stand": (self._initStandTask, self._execStandTask),
            "drop": (self._initDropTask, self._execDropTask),
            "kinematic-walk": (
                self._initKinematicWalkTask,
                self._execKinematicWalkTask,
            ),
            "dynamic-walk": (
                self._initDynamicWalkTask,
                self._execDynamicWalkTask,
            ),
        }

        # Thread handling
        self.low_cmd_write_thread = None

        self.crc = CRC()

        # Initialize `low_cmd`
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for cmd in self.low_cmd.motor_cmd:
            cmd.mode = 0x01  # (PMSM) mode
            cmd.q = PosStopF
            cmd.kp = 0
            cmd.dq = VelStopF
            cmd.kd = 0
            cmd.tau = 0

        # Create publisher
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # Create subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._lowStateMessageHandler, 1)

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
        if self.run_mapper:
            self.mapper = subprocess.Popen(self.mapper_command)

        self.low_cmd_write_thread = RecurrentThread(
            interval=self.dt,
            target=self._lowCmdWrite,
            name="writebasiccmd",  # TODO: dt=0.009 works
        )
        self.low_cmd_write_thread.Start()

    def join(self):
        Event().wait()

    def stop(self):
        # Stop every thread calling into the controller, then let garbage
        # collection free it, since it is left out of collections while it runs
        self._stopDynamicWalkSolver()
        self.low_cmd_write_thread.Wait()
        self.lowstate_subscriber.Close()
        if self.mapper is not None:
            self.mapper.terminate()
            self.mapper.wait()
        self.shared_poses.close(unlink=True)
        if self.shared_map is not None:
            self.shared_map.close()
        gc.unfreeze()

    def _stopDynamicWalkSolver(self):
        # Stop the dynamic walk solver, if it is running
        self.solver_running = False
        if self.solver_thread is not None:
            self.solver_thread.join()
            self.solver_thread = None

    def set_task(self, task):
        if task not in self.task_map:
            raise ValueError("Invalid task")

        if self.fallback_start is not None:
            print(f"Ignoring the {task} task during the fallback")
            return

        self._stopDynamicWalkSolver()

        # Initialize task via task initializer function
        self.task_map[task][0]()
        # Set task as the current task
        self.task = task

    # Private methods
    def _lowStateMessageHandler(self, msg: LowState_):
        filtering = self.task == "dynamic-walk"
        with self.state_lock:
            self.low_state = msg
            self.low_state_time = time.time()
            if filtering:
                pos, vel = self._measuredState()
                acc = np.array(msg.imu_state.accelerometer)

        # Update the state filter with every message, and share its estimate,
        # the feet it takes to be on the ground and whether the robot is
        # jumping with the map process
        if filtering:
            now = time.time()
            x, R = self.filter_state(pos, vel, acc, now)
            self.shared_poses.write(
                now,
                x[:3],
                np.roll(dwm.rotation_to_quat(R), 1),
                pos[7:],
                self.stance(),
                self.jumping(),
            )

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
                self.set_task(self.walk_task)
            else:
                self.task_map[self.task][1]()

            # Publish generated commands
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

    def _command(self, q, dq, kp, kd, tau):
        # Sets the motor commands, each from a value per motor (robot order)
        # or one for all of them
        q, dq, kp, kd, tau = np.broadcast_arrays(q, dq, kp, kd, tau)
        for i in range(12):
            cmd = self.low_cmd.motor_cmd[i]
            cmd.q = q[i]
            cmd.dq = dq[i]
            cmd.kp = kp[i]
            cmd.kd = kd[i]
            cmd.tau = tau[i]

    def _moveJoints(self, q):
        # Commands the joints to q (robot order) with the stand gains
        self._command(q, 0.0, self.Kp, self.Kd, 0.0)

    def _initStandTask(self):
        self.first_run = True
        self.percent_1 = 0.0
        self.percent_2 = 0.0
        self.percent_3 = 0.0

    def _execStandTask(self):
        # On first run, copy current motor state
        if self.first_run:
            self.start_pos = [
                self.low_state.motor_state[i].q for i in range(12)
            ]
            self.first_run = False

        # On sub-move 1, interpolate between initial and stand-ground positions
        self.percent_1 = min(self.percent_1 + self.dt / 0.5, 1)
        if self.percent_1 < 1:
            self._moveJoints(
                blend(self.start_pos, self.STAND_GROUND_POS, self.percent_1)
            )

        # On sub-move 2, interpolate between stand-ground and stance positions
        elif self.percent_2 < 1:
            self.percent_2 = min(self.percent_2 + self.dt / 2.0, 1)
            self._moveJoints(
                blend(self.STAND_GROUND_POS, self.STANCE_POS, self.percent_2)
            )

        # On sub-move 3, hold stance position
        elif self.percent_3 < 1:
            self.percent_3 = min(self.percent_3 + self.dt / 2.0, 1)
            self._moveJoints(self.STANCE_POS)

        # On finish, revert to None task
        else:
            self.task = None

    def _initDropTask(self):
        self.percent = 0.0

    def _execDropTask(self):
        # Interpolate between stance and ground positions
        if self.percent < 1:
            self.percent = min(self.percent + self.dt / 2.5, 1)
            self._moveJoints(
                blend(self.STANCE_POS, self.DROP_POS, self.percent)
            )

        # On finish, revert to None task
        else:
            self.task = None

    def _initKinematicWalkTask(self):
        self.step = 0
        self.stance_feet = []

        self.model, self.mpc_controller, self.update_func = (
            kinematic_walk_model.get_controller(self._measuredState()[0])
        )

    def _execKinematicWalkTask(self):
        with self.state_lock:
            pos, vel = self._measuredState()

        # Estimate the base height from the feet in stance, which are on the
        # ground, using forward kinematics with the base at the origin
        config = kinematic_walk_model.go2_planner.config
        if self.stance_feet:
            local = pos.copy()
            local[:3] = 0.0
            height = -np.mean(
                [
                    self.model.forwardKinematics(
                        local, config.FRAME_NAMES[foot]
                    )[0][2]
                    for foot in self.stance_feet
                ]
            )
            # Offset height by some amount
            pos[2] = height + 0.05

        limits = config.ReferenceGeneration
        v_cmd = self.joystick.read(
            limits.max_vel_front,
            limits.max_vel_back,
            limits.max_vel_side,
            limits.max_vel_yaw,
        )
        self.step += 1

        p_ref = self.update_func(v_cmd, pos, vel, self.step)

        # The feet the references have in stance
        self.stance_feet = [foot for foot in p_ref if not p_ref[foot][0][1]]

        x, u = (v.toarray() for v in self.mpc_controller.step(pos))
        self._command(
            x[1][7:][self.JOINT_REORDER_MAP],
            u[1][6:][self.JOINT_REORDER_MAP],
            self.JOINT_Kps,
            self.JOINT_Kds,
            0.0,
        )

    def _initDynamicWalkTask(self):
        # Ground the walk steps on and the distances to its edges, flat until
        # the walk has read MAP_SETTLE_MAPS LIDAR maps. The walk's frame
        # starts over here, so the map process starts a new map, and maps
        # made before now are left out
        self.terrain = dwm.ssqpy.model.terrain.HeightmapTerrain(
            np.zeros((2, 2)), em.MAP_RESOLUTION
        )
        # Heights and corner of the latest map, which the terrain goes back to
        # once the walk uses the map again, and whether the walk leaves the
        # map out and takes the terrain to be flat
        self.terrain_heights = (np.zeros((2, 2)), 0.0, 0.0)
        self.flat_terrain = False
        self.edges = None
        self.terrain_time = time.time()
        self.maps_read = 0
        self.shared_poses.restart()

        (
            self.model,
            self.mpc_controller,
            self.update_func,
            self.plan_commands,
            self.command_torques,
            self.filter_state,
            self.flat_ground,
            self.jumping,
            self.stance,
        ) = dwm.get_controller(
            self._measuredState()[0],
            SIMULATION,
            self.terrain,
            self._edgeDistance,
        )

        # Leave everything made so far out of garbage collections, which
        # would otherwise pause every thread for several milliseconds
        # each time they go through all objects
        gc.freeze()

        # Latest plan with the time it started
        self.plan = None
        self.stance_error_commands = 0
        self.swing_error_commands = 0

        # Solve in a separate thread, so that commands keep going out and
        # the state keeps updating while it solves
        self.solver_running = True
        self.solver_thread = Thread(
            target=self._runDynamicWalkSolver, daemon=True
        )
        self.solver_thread.start()

    def _runDynamicWalkSolver(self):
        cores = performance_cores()
        if cores is not None:
            os.sched_setaffinity(0, cores)

        next_solve = time.time()
        rate_start = next_solve
        solve_times = []

        while self.solver_running:
            delay = next_solve - time.time()
            if delay > 0:
                time.sleep(delay)

            # Get generalized position and velocity vectors
            now = time.time()
            with self.state_lock:
                pos, vel = self._measuredState()
                forces = np.array(self.low_state.foot_force)

            # The simulator counts as reporting foot forces once any foot has
            # reported one
            if SIMULATION and forces.any():
                self.sim_foot_forces = True

            contacts = None
            if not SIMULATION or self.sim_foot_forces:
                contacts = list(
                    forces[self.FOOT_REORDER_MAP] > self.FOOT_CONTACT_FORCE
                )

            v_cmd = self.joystick.read(
                dwm.MAX_VEL_FRONT,
                dwm.MAX_VEL_BACK,
                dwm.MAX_VEL_SIDE,
                dwm.MAX_VEL_YAW,
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

            # The jump button jumps forward while the forward jump button is
            # held, and in place otherwise
            jump = self.JUMP_BUTTON in buttons
            forward = self.FORWARD_JUMP_BUTTON in buttons

            # Solve from the state predicted for when the plan starts
            start = now + dwm.PREDICT_TIME
            plan = self.plan

            try:
                self._updateTerrain()
                state = self.update_func(
                    v_cmd,
                    self.joystick.dpad_x,
                    self.joystick.dpad_y,
                    orientation,
                    jump,
                    self.GAIT_BUTTON in buttons,
                    pos,
                    vel,
                    contacts,
                    forward,
                )
                if plan is not None:
                    state = dwm.predict_state(
                        state, plan[1], now - plan[0], start - plan[0]
                    )
                x, u = self.mpc_controller.step(state)
                commands = self.plan_commands(x, u)
            except Exception as err:
                self._startFallback(f"the solver failed: {err}")
                return

            if self.show_rate:
                solve_times.append(time.time() - now)
                elapsed = time.time() - rate_start
                if elapsed >= self.RATE_PERIOD:
                    print(
                        f"MPC {len(solve_times) / elapsed:.1f} Hz, solve "
                        f"{1000 * np.mean(solve_times):.2f} ms mean, "
                        f"{1000 * np.max(solve_times):.2f} ms max"
                    )
                    rate_start = time.time()
                    solve_times = []

            # Apply the plan as soon as it starts, rather than at the next
            # command
            delay = start - time.time()
            if delay > 0:
                time.sleep(delay)
            self.plan = (start, commands)
            self._lowCmdWrite()

            # Keep a fixed solve rate, but skip stages that are already late
            next_solve = max(next_solve + dwm.MPC_DT, time.time())

    def _updateTerrain(self):
        # Takes the latest LIDAR map into the walk's terrain if it is new.
        # Only the solver thread uses the terrain, so it is replaced here
        # between solves. While the walk leaves the map out, the terrain is
        # flat at the height it gives instead, with no edges, and it
        # goes back to the latest map once the walk uses the map again
        ground = self.flat_ground()
        if ground is not None and not self.flat_terrain:
            self.terrain.set_heights(np.full((2, 2), ground))
            self.flat_terrain = True
        elif ground is None and self.flat_terrain:
            self.terrain.set_heights(*self.terrain_heights)
            self.flat_terrain = False

        if not self.use_map:
            return
        if self.shared_map is None:
            try:
                self.shared_map = em.SharedMap(em.MAP_NAME)
            except FileNotFoundError:
                return

        if self.shared_map.time() <= self.terrain_time:
            return
        latest = self.shared_map.read()
        if latest is None:
            return

        heights, edges, x0, y0, self.terrain_time = latest
        self.maps_read += 1
        if self.maps_read < self.MAP_SETTLE_MAPS:
            return
        self.terrain_heights = (heights + dwm.FOOT_POINT_HEIGHT, x0, y0)
        if not self.flat_terrain:
            self.terrain.set_heights(*self.terrain_heights)
        self.edges = (edges, x0, y0)

    def _edgeDistance(self, x, y):
        # Distance from (x, y) to the nearest edge in the latest map, taking
        # there to be none outside of it or while the terrain is flat
        if self.edges is None or self.flat_terrain:
            return np.inf

        edges, x0, y0 = self.edges
        i = round((x - x0) / em.MAP_RESOLUTION)
        j = round((y - y0) / em.MAP_RESOLUTION)
        if not (0 <= i < edges.shape[0] and 0 <= j < edges.shape[1]):
            return np.inf
        return edges[i, j]

    def _execDynamicWalkTask(self):
        # Keep the previous commands until the first plan is ready
        plan = self.plan
        if plan is None:
            return

        with self.state_lock:
            pos, vel = self._measuredState()
            state_time = self.low_state_time

        # Play the latest plan back by the time elapsed since it started, ahead
        # by the time the motors take to act on the commands
        now = time.time()
        elapsed = now - plan[0]
        lead = dwm.SIM_COMMAND_LEAD if SIMULATION else dwm.COMMAND_LEAD
        q, dq = dwm.playback(plan[1], elapsed + lead)
        tau = self.command_torques(plan[1], elapsed + lead, pos, vel)

        # Joints off the plan as it is now, counted separately for the legs
        # the plan has in stance and the swinging ones
        planned_q, _ = dwm.playback(plan[1], elapsed)
        error = planned_q - pos[7:]
        worst = np.argmax(np.abs(error))
        stages = plan[1][2]
        k = min(int(max(elapsed, 0.0) / dwm.MPC_DT), len(stages) - 1)
        stance = np.repeat([name in stages[k] for name in dwm.FOOT_NAMES], 3)
        off = np.abs(error) > self.MAX_TRACKING_ERROR
        self.stance_error_commands = (
            self.stance_error_commands + 1 if off[stance].any() else 0
        )
        self.swing_error_commands = (
            self.swing_error_commands + 1 if off[~stance].any() else 0
        )

        # Cosine of the angle between the measured and planned base tilts,
        # comparing the world z axis as seen from the base, which ignores yaw
        planned = dwm.plan_state(plan[1][0], elapsed)
        cos_tilt = (
            dwm.quat_to_rotation(pos[3:7])[2]
            @ dwm.quat_to_rotation(planned[3:7])[2]
        )

        reason = None
        if elapsed > self.MAX_PLAN_AGE:
            reason = "the solver stalled"
        elif now - state_time > self.MAX_STATE_AGE:
            reason = "the state stopped updating"
        elif not np.all(np.isfinite(np.hstack((q, dq, tau)))):
            reason = "the commands are not finite"
        elif (
            self.stance_error_commands >= self.TRACKING_ERROR_COMMANDS
            or self.swing_error_commands >= self.SWING_TRACKING_ERROR_COMMANDS
        ):
            leg = ["FL", "FR", "RL", "RR"][worst // 3]
            joint = ["hip", "thigh", "calf"][worst % 3]
            reason = (
                f"the joints stopped following the plan ({leg} {joint} at "
                f"{pos[7 + worst]:.2f} rad, planned {planned_q[worst]:.2f} "
                "rad)"
            )
        elif cos_tilt < np.cos(self.MAX_TILT):
            reason = "the base tilted too far from the plan"

        if reason is not None:
            self._startFallback(reason)
            self._execFallbackTask()
            return

        if SIMULATION:
            kp, kd = dwm.SIM_JOINT_KP, dwm.SIM_JOINT_KD
        else:
            kp, kd = dwm.JOINT_KP, dwm.JOINT_KD
        reorder = self.JOINT_REORDER_MAP
        self._command(q[reorder], dq[reorder], kp, kd, tau[reorder])

    def _startFallback(self, reason):
        with self.cmd_lock:
            if self.fallback_start is not None:
                return

            print(f"Falling back since {reason}")
            self.solver_running = False
            self.start_pos = [
                self.low_state.motor_state[i].q for i in range(12)
            ]
            self.fallback_start = time.time()

    def _execFallbackTask(self):
        elapsed = time.time() - self.fallback_start

        # Move the joints to the stance position, then to the lying position
        if elapsed < self.FALLBACK_STAND_TIME:
            percent = elapsed / self.FALLBACK_STAND_TIME
            self._moveJoints(blend(self.start_pos, self.STANCE_POS, percent))
        elif elapsed < self.FALLBACK_STAND_TIME + self.FALLBACK_DROP_TIME:
            percent = (
                elapsed - self.FALLBACK_STAND_TIME
            ) / self.FALLBACK_DROP_TIME
            self._moveJoints(blend(self.STANCE_POS, self.DROP_POS, percent))

        # Once lying down, damp the joints
        else:
            for cmd in self.low_cmd.motor_cmd[:12]:
                cmd.dq = 0.0
                cmd.kp = 0.0
                cmd.kd = self.DAMPING_KD
                cmd.tau = 0.0

    def _measuredState(self):
        # The configuration and velocity from the latest state message, with
        # the joints reordered to the model's order
        state = self.low_state
        quat = np.roll(state.imu_state.quaternion, -1)
        quat /= np.linalg.norm(quat)
        motors = [state.motor_state[j] for j in self.JOINT_REORDER_MAP]
        config = np.hstack((self.BASE_POS, quat, [m.q for m in motors]))
        vel = np.hstack(
            (np.zeros(3), state.imu_state.gyroscope, [m.dq for m in motors])
        )
        return config, vel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Go2 walking MPC controller driven with a gamepad"
    )

    parser.add_argument(
        "-k",
        "--kinematic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run the kinematic walk instead of the dynamic walk",
    )

    parser.add_argument(
        "-s",
        "--simulation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to control the unitree_mujoco simulator instead of the "
        "robot",
    )

    parser.add_argument(
        "-i",
        "--interface",
        type=str,
        default=None,
        help=f"The network interface to use (default: {SIMULATION_INTERFACE} "
        f"when simulating, {ROBOT_INTERFACE} otherwise)",
    )

    parser.add_argument(
        "-v",
        "--view",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to show the LIDAR map in a Rerun viewer",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Rerun .rrd file to save the LIDAR map to",
    )

    parser.add_argument(
        "--map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the dynamic walk uses the LIDAR map, rather than taking "
        "the ground to be flat. The map is still made when shown or saved",
    )

    parser.add_argument(
        "-r",
        "--rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to print the dynamic walk's MPC rate and solve times "
        "every few seconds",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    SIMULATION = args.simulation
    interface = args.interface or (
        SIMULATION_INTERFACE if SIMULATION else ROBOT_INTERFACE
    )

    ChannelFactoryInitialize(0, interface)

    custom = CustomGo2Controller(
        interface,
        "kinematic-walk" if args.kinematic else "dynamic-walk",
        args.view,
        args.save,
        args.map,
        args.rate,
    )
    custom.start()

    try:
        custom.set_task("stand")
        time.sleep(3.0)

        custom.set_task(custom.walk_task)
        custom.join()
    except KeyboardInterrupt:
        pass
    finally:
        custom.stop()
