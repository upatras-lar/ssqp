import numpy as np

# Helpers
# -------


def yaw_to_Rz(yaw):
    return np.array(
        (
            (np.cos(yaw), -np.sin(yaw), 0.0),
            (np.sin(yaw), np.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def euler_to_quat(roll, pitch, yaw):
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return np.array([x, y, z, w])


def quat_to_yaw(quat):
    qx, qy, qz, qw = quat
    return np.arctan2(2.0 * (qz * qw + qx * qy), -1.0 + 2.0 * (qw**2 + qx**2))


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


def quat_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix


def quat_integrate(q, omega_dt):
    """Integrate quaternion q by angular-velocity * dt (expressed in world)."""
    angle = np.linalg.norm(omega_dt)
    if angle < 1e-10:
        return q / np.linalg.norm(q)
    axis = omega_dt / angle
    dq = np.array(
        [
            np.sin(angle / 2) * axis[0],
            np.sin(angle / 2) * axis[1],
            np.sin(angle / 2) * axis[2],
            np.cos(angle / 2),
        ]
    )
    return quat_mul(q, dq)


def skew(v):
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
