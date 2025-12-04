import numpy as np


def rotate_vector_inverse_rpy(roll, pitch, yaw, vector):
    """
    Rotate a vector by the inverse of the given roll, pitch, and yaw angles.

    Parameters:
    roll (float): The roll angle in radians.
    pitch (float): The pitch angle in radians.
    yaw (float): The yaw angle in radians.
    vector (np.ndarray): The 3D vector to be rotated.

    Returns:
    np.ndarray: The rotated 3D vector.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R_z @ R_y @ R_x).T @ vector


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate (inverse rotation).

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a quaternion.

    Args:
        q: Quaternion [w, x, y, z]
        v: 3D vector [x, y, z]

    Returns:
        Rotated vector [x, y, z]
    """
    # Convert vector to quaternion [0, x, y, z]
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)

    # Compute q * v * q_conjugate
    result = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))

    # Extract vector part
    return result[1:4]


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation (first 2 columns flattened).

    Args:
        R: 3x3 rotation matrix

    Returns:
        6D vector (first 2 columns of R flattened)
    """
    return R[:, :2].reshape(-1)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return a normalized copy of the quaternion."""

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def quat_from_yaw(yaw: float) -> np.ndarray:
    """Construct a quaternion representing a pure yaw rotation."""

    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def quat_yaw(q: np.ndarray) -> float:
    """Extract the yaw angle from a quaternion."""

    w, x, y, z = quat_normalize(q)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix back to a quaternion [w, x, y, z]."""

    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat_normalize(quat)
