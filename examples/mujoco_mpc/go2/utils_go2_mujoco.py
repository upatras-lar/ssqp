import numpy as np


def quat_wxyz_to_xyzw(wxyz):
    xyzw = np.zeros(4)
    xyzw[:3] = wxyz[1:]
    xyzw[3] = wxyz[0]
    return xyzw


def quat_xyzw_to_wxyz(xyzw):
    wxyz = np.zeros(4)
    wxyz[0] = xyzw[3]
    wxyz[1:] = xyzw[:3]
    return wxyz


def ssqp_to_mujoco_state(ssqp_state):
    mujoco_state = ssqp_state.copy()
    mujoco_state[3:7] = quat_xyzw_to_wxyz(ssqp_state[3:7])
    # mujoco_state[7:10] = ssqp_state[10:13]
    # mujoco_state[10:13] = ssqp_state[7:10]
    # mujoco_state[13:16] = ssqp_state[16:19]
    # mujoco_state[16:19] = ssqp_state[13:16]
    # mujoco_state[25:28] = ssqp_state[28:31]
    # mujoco_state[28:31] = ssqp_state[25:28]
    # mujoco_state[31:34] = ssqp_state[34:37]
    # mujoco_state[34:37] = ssqp_state[31:34]
    return mujoco_state


def ssqp_to_mujoco_control(ssqp_control):
    mujoco_control = ssqp_control.copy()
    # mujoco_control[6:9] = ssqp_control[9:12]
    # mujoco_control[9:12] = ssqp_control[6:9]
    # mujoco_control[12:15] = ssqp_control[15:18]
    # mujoco_control[15:18] = ssqp_control[12:15]
    return mujoco_control


def mujoco_to_ssqp_state(mujoco_state):
    ssqp_state = mujoco_state.copy()
    ssqp_state[3:7] = quat_wxyz_to_xyzw(mujoco_state[3:7])
    # ssqp_state[7:10] = mujoco_state[10:13]
    # ssqp_state[10:13] = mujoco_state[7:10]
    # ssqp_state[13:16] = mujoco_state[16:19]
    # ssqp_state[16:19] = mujoco_state[13:16]
    # ssqp_state[25:28] = mujoco_state[28:31]
    # ssqp_state[28:31] = mujoco_state[25:28]
    # ssqp_state[31:34] = mujoco_state[34:37]
    # ssqp_state[34:37] = mujoco_state[31:34]
    return ssqp_state


def ssqp_to_mujoco_state_kin(ssqp_config):
    mujoco_config = ssqp_config.copy()
    mujoco_config[3:7] = quat_xyzw_to_wxyz(ssqp_config[3:7])
    # mujoco_config[7:10] = ssqp_config[10:13]
    # mujoco_config[10:13] = ssqp_config[7:10]
    # mujoco_config[13:16] = ssqp_config[16:19]
    # mujoco_config[16:19] = ssqp_config[13:16]
    return mujoco_config


def ssqp_to_mujoco_vel_kin(ssqp_vel):
    mujoco_vel = ssqp_vel.copy()
    # mujoco_vel[6:9] = ssqp_vel[9:12]
    # mujoco_vel[9:12] = ssqp_vel[6:9]
    # mujoco_vel[12:15] = ssqp_vel[15:18]
    # mujoco_vel[15:18] = ssqp_vel[12:15]
    return mujoco_vel


def ssqp_to_mujoco_ctrl_kin(ssqp_ctrl):
    mujoco_ctrl = ssqp_ctrl.copy()
    mujoco_ctrl[:3] = ssqp_ctrl[3:6]
    mujoco_ctrl[3:6] = ssqp_ctrl[:3]
    mujoco_ctrl[6:9] = ssqp_ctrl[9:]
    mujoco_ctrl[9:] = ssqp_ctrl[6:9]
    return mujoco_ctrl


def mujoco_to_ssqp_state_kin(mujoco_config):
    ssqp_config = mujoco_config.copy()
    ssqp_config[3:7] = quat_wxyz_to_xyzw(mujoco_config[3:7])
    # ssqp_config[7:10] = mujoco_config[10:13]
    # ssqp_config[10:13] = mujoco_config[7:10]
    # ssqp_config[13:16] = mujoco_config[16:19]
    # ssqp_config[16:19] = mujoco_config[13:16]
    return ssqp_config
