import ssqpy
from pylibfranka import (
    RealtimeConfig,
    Robot,
    Torques,
)
import numpy as np
import argparse
import time

import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from sys import getsizeof

import models
from joystick import Controller

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control script moving the end-effector of the real panda "
        "robot to a certain location"
    )

    subparsers = parser.add_subparsers(
        dest="model", help="The model to execute"
    )

    subparsers.add_parser(
        "reset", help="Resets the Panda's configuration to the initial"
    )

    parser_target = subparsers.add_parser(
        "target", help="Moves the Panda's end-effector to a target position"
    )
    parser_target.add_argument(
        "--pos",
        nargs=3,
        type=float,
        default=(0.46, 0.35, 0.0),
        help="The target end-effector position",
    )
    parser_target.add_argument(
        "--joystick",
        default=False,
        action="store_true",
        help="Whether to accept joystick control",
    )

    parser_circle = subparsers.add_parser(
        "circle",
        help="Performs a circle in the yz-plane with the Panda's end-effector",
    )
    parser_circle.add_argument(
        "--constrain",
        type=bool,
        default=False,
        help="Whether to add the inscribed square constraints",
    )

    parser.add_argument(
        "-r",
        "--robot-ip",
        type=str,
        required=True,
        help="The robot IP",
    )

    parser.add_argument(
        "-m",
        "--dynamics-mode",
        choices=["inverse", "forward"],
        default="forward",
        help="The dynamics mode to use",
    )

    parser.add_argument(
        "-a",
        "--interpolation-alpha",
        type=float,
        default=0.999,
        help="The interpolation alpha parameter",
    )

    return parser.parse_args()


args = parse_args()

if args.model is None:
    args.model = "reset"

print(f"Setting model as: {args.model}")

state_lock = multiprocessing.Lock()
state_memory = SharedMemory(create=True, size=14 * getsizeof(np.double))

controls_lock = multiprocessing.Lock()
controls_memory = SharedMemory(create=True, size=7 * getsizeof(np.double))


def MPC_process(state_memory_name, controls_memory_name):
    state_memory = SharedMemory(name=state_memory_name)
    controls_memory = SharedMemory(name=controls_memory_name)

    mode_map = {
        "inverse": ssqpy.model.DynamicsMode.Inverse,
        "forward": ssqpy.model.DynamicsMode.Forward,
    }

    get_model_func = None
    model_func_params = []
    if args.model == "target":
        get_model_func = models.get_target_model
        model_func_params.append(args.pos)
    elif args.model == "reset":
        get_model_func = models.get_reset_model
    elif args.model == "circle":
        get_model_func = models.get_circle_model
        model_func_params.append(args.constrain)

    model, callback = get_model_func(
        mode_map[args.dynamics_mode], *model_func_params
    )

    nq = model.getnq()
    nv = model.getnv()

    ssqpy.setSilentMode()

    ssqp_params = ssqpy.solvers.ssqpParams()
    ssqp_params.tolerance = 1e-2

    admm_params = ssqpy.solvers.admmParams()
    admm_params.abs_tolerance = 1e-3
    admm_params.rel_tolerance = 1e-3
    admm_params.init_rho = 0.1
    admm_params.warm_start = False

    ssqp_params.admmParams = admm_params

    mpc = ssqpy.solvers.MPC(model, ssqp_params, sqp_iters=4, qp_iters=125)

    if args.model == "target" and args.joystick:
        controller = Controller()

    stage = 0

    if args.model == "target":
        current_target = np.array(args.pos)

    try:
        while True:
            # Get state data from hardware
            with state_lock:
                state_data = np.copy(
                    np.ndarray((14,), dtype=np.double, buffer=state_memory.buf)
                )

            state = np.zeros(nq + nv)
            state[:7] = state_data[:7]
            state[9:16] = state_data[7:]

            controls = mpc.step(state)[0].stage(1)

            # Give controls to hardware process
            with controls_lock:
                controls_buffer = np.ndarray(
                    (7,), dtype=np.double, buffer=controls_memory.buf
                )
                controls_buffer[:] = controls[:7]

            stage += 1
            if args.model != "target" or not args.joystick:
                callback(stage)
            else:
                keys = controller.read()
                if keys[0] < 128:
                    current_target[1] -= 0.01
                elif keys[0] > 128:
                    current_target[1] += 0.01

                if keys[1] < 128:
                    current_target[0] -= 0.01
                elif keys[1] > 128:
                    current_target[0] += 0.01

                if keys[2]:
                    current_target[2] += 0.01

                if keys[3]:
                    current_target[2] -= 0.01

                callback(np.array(current_target))

            time.sleep(0.05)
    except KeyboardInterrupt:
        state_memory.close()
        controls_memory.close()


def hardware_process(state_memory_name, controls_memory_name):
    state_memory = SharedMemory(name=state_memory_name)
    controls_memory = SharedMemory(name=controls_memory_name)

    # Initialize hardware
    robot = Robot(args.robot_ip)

    # Set default behavior
    robot.set_collision_behavior(
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    )

    # Create a model instance from robot
    model = robot.load_model()

    prev_controls = None

    active_control = robot.start_torque_control()

    target_control = None
    current_control = None

    # Compliance parameters
    joint_stiffness = [10.0] * 7
    joint_damping = [0.8 * np.sqrt(k) for k in joint_stiffness]

    period = 0.0001

    try:
        while True:
            t0 = time.time()
            robot_state, _ = active_control.readOnce()

            state = np.hstack((robot_state.q, robot_state.dq))

            coriolis = np.array(model.coriolis(robot_state))
            q = np.array(robot_state.q)
            dq = np.array(robot_state.dq)

            # Give state to MPC
            with state_lock:
                state_buffer = np.ndarray(
                    (14,), dtype=np.double, buffer=state_memory.buf
                )
                state_buffer[:] = state[:]

            if target_control is None:
                target_control = q.copy()
                current_control = target_control.copy()

            # Receive controls from MPC
            with controls_lock:
                target_control = np.copy(
                    np.ndarray(
                        (7,), dtype=np.double, buffer=controls_memory.buf
                    )
                )

            current_control = (
                args.interpolation_alpha * current_control
                + (1.0 - args.interpolation_alpha) * target_control
            )

            # Compute error to desired equilibrium joint configuration
            position_error = q - current_control

            # Compute joint-space impedance control
            tau_task = np.zeros(7)
            for i in range(7):
                tau_task[i] = (
                    -joint_stiffness[i] * position_error[i]
                    - joint_damping[i] * dq[i]
                )

            # Add coriolis compensation
            tau_d = tau_task + coriolis

            command = Torques(tau_d.tolist())
            command.motion_finished = False
            active_control.writeOnce(command)

            next_time = t0 + period
            sleep_time = next_time - time.time()
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        state_memory.close()
        controls_memory.close()


mpc_process = multiprocessing.Process(
    target=MPC_process,
    args=(state_memory.name, controls_memory.name),
    daemon=True,
)
mpc_process.start()

hardware_process(state_memory.name, controls_memory.name)

mpc_process.join()

state_memory.close()
state_memory.unlink()

controls_memory.close()
controls_memory.unlink()
