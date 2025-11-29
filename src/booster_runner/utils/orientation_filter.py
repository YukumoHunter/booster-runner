import numpy as np

from .rotate import quat_conjugate, quat_multiply, quat_rotate_vector


class OrientationFilter:
    """Simple complementary filter that fuses gyro and accel readings."""

    def __init__(
        self,
        dt: float,
        accel_correction_gain: float = 0.05,
        gravity_magnitude: float = 9.81,
    ) -> None:
        self.default_dt = float(dt)
        self.accel_correction_gain = accel_correction_gain
        self.gravity_magnitude = gravity_magnitude
        self._gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._initialized = False

    @property
    def quaternion(self) -> np.ndarray:
        return self._quat.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self, quat: np.ndarray) -> None:
        self._quat = self._normalize(quat)
        self._initialized = True

    def update(
        self,
        gyro: np.ndarray,
        accel: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        gyro = np.asarray(gyro, dtype=np.float32)
        gyro_corrected = gyro.copy()
        dt_step = self.default_dt if dt is None else float(dt)

        if accel is not None:
            accel = np.asarray(accel, dtype=np.float32)
            accel_norm = np.linalg.norm(accel)
            if (
                accel_norm > 1e-6
                and abs(accel_norm - self.gravity_magnitude) <= self.gravity_magnitude
            ):
                accel_dir = accel / accel_norm
                gravity_estimate = quat_rotate_vector(
                    quat_conjugate(self._quat), self._gravity_world
                )
                error = np.cross(gravity_estimate, accel_dir)
                gyro_corrected += self.accel_correction_gain * error

        delta_q = np.array([1.0, *(0.5 * gyro_corrected * dt_step)], dtype=np.float32)
        delta_q = self._normalize(delta_q)
        self._quat = self._normalize(quat_multiply(self._quat, delta_q))
        self._initialized = True
        return self.quaternion

    def _normalize(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return quat / norm
