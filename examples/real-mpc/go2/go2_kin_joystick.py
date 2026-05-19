import time
import numpy as np
import pinocchio as pin

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient

import kinematic_walk_model

import sys

sys.path.insert(0, "../../mujoco-mpc/go2")

from go2_controllers.joystick import Controller

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

    def __init__(self):
        self.Kp = 40.0
        self.Kd = 1.6

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.current_base_pos = kinematic_walk_model.go2_kinematic.init_config[
            :3
        ]
        self.current_base_vel = np.zeros(3)

        # Position before starting stand move
        self._stand_ground_pos = [
            0.0,
            1.36,
            -2.65,
            0.0,
            1.36,
            -2.65,
            -0.2,
            1.36,
            -2.65,
            0.2,
            1.36,
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

            self.mpc_controller, self.update_func = (
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

        # Create subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._lowStateMessageHandler, 10)

        """
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
        """

    def start(self):
        self.low_cmd_write_thread = RecurrentThread(
            interval=0.002, target=self._lowCmdWrite, name="writebasiccmd"
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
        self.low_state = msg

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
        self.percent_1 += 1.0 / 500.0
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
            self.percent_2 += 1.0 / 500.0
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
            self.percent_3 += 1.0 / 500.0
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
        self.percent_1 += 1.0 / 500.0
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
            self.percent_2 += 1.0 / 900.0
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
        # TODO: Find a way to get the correct generalized vectors
        pos = self._getGeneralizedConfiguration(self.current_base_pos)
        vel = self._getGeneralizedVelocity(self.current_base_vel)

        v_cmd = self.joystick.read()
        self.step += 1

        self.update_func(
            v_cmd,
            pos,
            vel,
            self.step,
        )

        next = [x.toarray() for x in self.mpc_controller.step(pos)]

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = next[0][1][
                7 + self.JOINT_REORDER_MAP[i]
            ]
            self.low_cmd.motor_cmd[i].dq = next[1][1][
                6 + self.JOINT_REORDER_MAP[i]
            ]
            self.low_cmd.motor_cmd[i].kp = 40.0
            self.low_cmd.motor_cmd[i].kd = 1.0
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

        return np.hstack((np.zeros(3), ang_vel, joint_vel))


if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp5s0")

    custom = CustomGo2Controller()
    custom.start()
    custom.set_task("stand")

    while True:
        time.sleep(4.0)
        custom.set_task("kinematic-walk")
        time.sleep(20.0)
        break
