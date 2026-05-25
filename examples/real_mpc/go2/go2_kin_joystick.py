import time
import numpy as np
from threading import Lock

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient

from rko_lio.config import PipelineConfig
from rko_lio.lio_pipeline import LIOPipeline

import kinematic_walk_model

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

    def __init__(self, max_vel_front, max_vel_back, max_vel_side, max_vel_yaw):
        self.max_vel_front = max_vel_front
        self.max_vel_back = max_vel_back
        self.max_vel_side = max_vel_side
        self.max_vel_yaw = max_vel_yaw

        self.Kp = 40.0
        self.Kd = 1.6
        self.dt = 0.01

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.current_base_pos = (
            kinematic_walk_model.go2_kinematic.config.INIT_BASE_POS
        )
        self.current_base_vel = np.zeros(3)

        self.state_lock = Lock()

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
            self.joystick = Controller()
            self.step = 0

            self.model, self.mpc_controller, self.update_func = (
                kinematic_walk_model.get_controller(
                    self._getGeneralizedConfiguration(self.current_base_pos)
                )
            )

        self.task_map = {
            None: (lambda obj: None, lambda: None),
            "stand": (stand_task, self._execStandTask),
            "drop": (drop_task, self._execDropTask),
            "kinematic-walk": (
                kinematic_walk_task,
                self._execKinematicWalkTask,
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

        # Create LIO
        self.lio_config = PipelineConfig()
        self.lio_config.extrinsic_imu2base = None
        self.lio_config.extrinsic_lidar2base = None

        self.lio_pipeline = LIOPipeline(self.lio_config)

        # Create subscribers
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._lowStateMessageHandler, 1)

        self.lidar_subscriber = ChannelSubscriber(
            "rt/utlidar/cloud", PointCloud2_
        )
        self.lidar_subscriber.Init(self._pointCloudMessageHandler, 1)

        """
        # Feet that are in contact (according to Kinematic model)
        self.stance_feet = []
        # Estimated base height
        self.height = None
        # Previous linear velocity
        self.prev_lin_vel = np.zeros(3)
        # Alpha parameter for accelerometer - kinematic velocity estimate
        self.vel_alpha = 0.99
        """

        # self.sportmode_subscriber = ChannelSubscriber(
        #     "rt/sportmodestate", SportModeState_
        # )
        # self.sportmode_subscriber.Init(self._sportStateMessageHandler, 1)

        self.sc = SportClient()
        self.sc.SetTimeout(0.2)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(0.2)
        self.msc.Init()

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

    def set_task(self, task):
        if task not in self.task_map.keys():
            raise ValueError("Invalid task")

        # Initialize task via task initializer function
        self.task_map[task][0](self)
        # Set task as the current task
        self.task = task

    # Private methods
    def _lowStateMessageHandler(self, msg: LowState_):
        self.state_lock.acquire()
        self.low_state = msg
        self.state_lock.release()

        self.lio_pipeline.add_imu(
            time.time(),
            msg.imu_state.accelerometer,
            msg.imu_state.gyroscope,
        )

        """
        if not self.stance_feet:
            return

        # Estimate body height
        velocity = []

        for motor in msg.motor_state[:12]:
            velocity.append(motor.dq)

        velocity = np.array(velocity)

        # Get configuration, assuming base is at (0.0, 0.0, 0.0)
        generalized_configuration = self._getGeneralizedConfiguration(
            np.zeros(3)
        )

        # Use forward kinematics to get position of each foot that is on the
        # ground, given base is at (0.0, 0.0, 0.0)
        feet_positions = []
        for foot in self.stance_feet:
            feet_positions.append(
                self.model.forwardKinematics(
                    generalized_configuration,
                    kinematic_walk_model.go2_kinematic.config.FRAME_NAMES[
                        foot
                    ],
                )[0]
            )

        # Since stance feet are at (0.0, 0.0, 0.0), the base will be at the
        # opposite height that forward kinematics calculated
        # So average the (opposite) height of each ground foot
        self.height = 0
        for pos in feet_positions:
            self.height -= pos[2]

        self.height /= len(feet_positions)
        # Offset height by some amount
        self.height += 0.05

        # TODO
        # Estimate linear velocity through Kinematics
        ang_vel = np.array(msg.imu_state.gyroscope)
        accel_body = np.array(msg.imu_state.accelerometer)

        pure_accel = accel_body

        lin_vel_kin = np.zeros(3)

        for foot in self.stance_feet:
            frame_name = kinematic_walk_model.go2_kinematic.config.FRAME_NAMES[
                foot
            ]

            v_foot_rel = (
                self.model.frameJacobian(
                    generalized_configuration, frame_name
                )[:3, 6:]
                @ velocity
            )

            lin_vel_kin += -v_foot_rel

        lin_vel_kin /= len(self.stance_feet)

        v_integrated = self.prev_lin_vel + pure_accel * self.dt
        lin_vel = (
            self.vel_alpha * v_integrated + (1 - self.vel_alpha) * lin_vel_kin
        )

        # Track history for the next iteration loop
        self.prev_lin_vel = lin_vel.copy()

        print(f"Estimated linear velocity: {lin_vel}")

        self.state_lock.acquire()
        self.current_base_pos = np.hstack((np.zeros(2), self.height))
        self.current_base_vel = np.zeros(3)
        self.state_lock.release()
        """

    def _pointCloudMessageHandler(self, msg: PointCloud2_):
        raw_bytes = bytes(msg.data)

        ROS_DTYPE = {
            1: np.int8,
            2: np.uint8,
            3: np.int16,
            4: np.uint16,
            5: np.int32,
            6: np.uint32,
            7: np.float32,
            8: np.float64,
        }

        # Sort fields by byte offset to lay them out in order.
        fields = sorted(msg.fields, key=lambda f: f.offset)

        dtype_list = []
        pos = 0
        for field in fields:
            # Insert explicit padding if there's a gap before this field.
            if field.offset > pos:
                dtype_list.append(
                    (f"_pad_{pos}", np.uint8, field.offset - pos)
                )
            dtype_list.append((field.name, ROS_DTYPE[field.datatype]))
            pos = field.offset + np.dtype(ROS_DTYPE[field.datatype]).itemsize

        # Pad out to point_step if there are trailing bytes.
        if pos < msg.point_step:
            dtype_list.append((f"_pad_{pos}", np.uint8, msg.point_step - pos))

        dtype = np.dtype(dtype_list)
        pc = np.frombuffer(raw_bytes, dtype=dtype)

        xyz = np.stack([pc["x"], pc["y"], pc["z"]], axis=1)  # (N, 3)

        header_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        timestamps = header_s + pc["time"].astype(np.float64)  # (N,)

        try:
            self.lio_pipeline.add_lidar(xyz, timestamps)
        except RuntimeError:
            return

        print(self.lio_pipeline.lio.pose())

    def _sportStateMessageHandler(self, msg: SportModeState_):
        self.state_lock.acquire()
        self.current_base_pos = np.hstack((np.zeros(2), msg.position[2]))
        self.current_base_vel = np.array(msg.velocity)
        self.state_lock.release()

        print(f"Got z-position: {msg.position[2]}")

    def _lowCmdWrite(self):
        if self.low_state is None:
            return

        # Execute task method
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
        self.percent_1 += self.dt
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
        elif self.percent_2 < 1:
            self.percent_2 += self.dt
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
            self.percent_3 += self.dt
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
        self.percent_1 += self.dt
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (
                    1 - self.percent_1
                ) * self.start_pos[i] + self.percent_1 * self._stance_pos[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # On sub-move 2, interpolate between stance and ground positions
        elif self.percent_2 < 1:
            self.percent_2 += self.dt / 2
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

        v_cmd = self.joystick.read(
            self.max_vel_front,
            self.max_vel_back,
            self.max_vel_side,
            self.max_vel_yaw,
        )
        self.step += 1

        p_ref = self.update_func(
            v_cmd,
            pos,
            vel,
            self.step,
        )

        """
        self.stance_feet = []
        for foot in p_ref:
            if not p_ref[foot][0][1]:  # if stance
                self.stance_feet.append(foot)
        """

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
    ChannelFactoryInitialize(0, "enp5s0")

    custom = CustomGo2Controller(
        kinematic_walk_model.go2_kinematic.config.ReferenceGeneration.max_vel_front,
        kinematic_walk_model.go2_kinematic.config.ReferenceGeneration.max_vel_back,
        kinematic_walk_model.go2_kinematic.config.ReferenceGeneration.max_vel_side,
        kinematic_walk_model.go2_kinematic.config.ReferenceGeneration.max_vel_yaw,
    )
    custom.start()

    custom.set_task("stand")
    time.sleep(6.0)
    # custom.set_task("kinematic-walk")
    # time.sleep(20.0)
    custom.set_task("drop")
    time.sleep(6.0)
