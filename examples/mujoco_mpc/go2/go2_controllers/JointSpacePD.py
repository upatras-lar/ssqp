import multiprocessing
import numpy as np
from time import time, sleep
from go2_controllers.joystick import Controller


class JointSpacePD:
    def __init__(self, mpc_controller, init_state, Kp=1.0, Kd=0.01):
        self.mpc_controller = mpc_controller
        self.Kp = Kp
        self.Kd = Kd

        self.last_mpc_sol = None
        self.mpc_traj_offset = 0

        self.state_q = multiprocessing.Queue()
        self.state_q.put(init_state)

        self.mpc_q = multiprocessing.Queue()
        self.mpc_proc = multiprocessing.Process(
            target=self._run_mpc, daemon=True
        )
        self.mpc_proc.start()

    def step(self, state):
        self.state_q.put(state)

        while self.last_mpc_sol is None and self.mpc_q.empty():
            sleep(0.1)

        if not self.mpc_q.empty():
            self.last_mpc_sol = self.mpc_q.get_nowait()
            self.mpc_traj_offset = 0

        control = (
            self.last_mpc_sol[1][self.mpc_traj_offset][6:18]
            + self.Kp
            * (
                self.last_mpc_sol[0][self.mpc_traj_offset + 1][7:19]
                - state[7:19]
            )
            + self.Kd
            * (
                self.last_mpc_sol[0][self.mpc_traj_offset + 1][25:]
                - state[25:]
            )
        )

        # if self.mpc_traj_offset < len(self.last_mpc_sol[1]) - 1:
        #     self.mpc_traj_offset += 1

        return control

    def _run_mpc(self):
        while True:
            try:
                while not self.state_q.empty():
                    state = self.state_q.get()
                next = [x.toarray() for x in self.mpc_controller.step(state)]
                self.mpc_q.put(next)
            except:
                sleep(0.01)


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

        # if self.mpc_traj_offset < len(self.last_mpc_sol[1]) - 2:
        #     self.mpc_traj_offset += 1

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

                # pre = time()

                next = [x.toarray() for x in self.mpc_controller.step(state)]

                # elapsed = time() - pre

                # if not step % 20:
                #     print(
                #         f"\rControl frequency: {(1 / elapsed):.2f}Hz\t", end=""
                #     )

                self.mpc_q.put(next)

            except Exception as exc:
                print(exc)
                sleep(0.01)
