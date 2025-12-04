"""Simple base pose filter that fuses IMU acceleration with zero-velocity updates."""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter


class BasePoseFilter:
    """Constant-acceleration Kalman filter with optional ZUPT corrections."""

    def __init__(
        self,
        dt: float,
        vel_meas_noise: float = 1e-3,
        bias_meas_noise: float = 1e-2,
        process_noise_pos: float = 1e-5,
        process_noise_vel: float = 1e-3,
        process_noise_bias: float = 1e-4,
    ) -> None:
        self.dt = float(dt)
        self.kf = KalmanFilter(dim_x=9, dim_z=3)

        # Constant velocity state transition.
        self.F = np.eye(9)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        self.kf.F = self.F

        self.Q = np.diag(
            [
                process_noise_pos,
                process_noise_pos,
                process_noise_pos,
                process_noise_vel,
                process_noise_vel,
                process_noise_vel,
                process_noise_bias,
                process_noise_bias,
                process_noise_bias,
            ]
        )

        self.H_vel = np.zeros((3, 9))
        self.H_vel[:, 3:6] = np.eye(3)
        self.R_vel = np.eye(3) * vel_meas_noise

        self.H_bias = np.zeros((3, 9))
        self.H_bias[:, 6:9] = np.eye(3)
        self.R_bias = np.eye(3) * bias_meas_noise

        self.reset()

    def reset(self) -> None:
        self.kf.x = np.zeros((9, 1))
        self.kf.P = np.eye(9) * 1e-3

    def set_state(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.kf.x[:3, 0] = np.asarray(position, dtype=np.float64)
        self.kf.x[3:6, 0] = np.asarray(velocity, dtype=np.float64)
        self.kf.x[6:9, 0] = 0.0

    def predict(self, lin_acc_w: np.ndarray) -> None:
        acc = np.asarray(lin_acc_w, dtype=np.float64).reshape(3, 1)
        # Remove bias estimate directly.
        acc -= self.kf.x[6:9]

        self.kf.x[:3] += self.kf.x[3:6] * self.dt + 0.5 * acc * (self.dt ** 2)
        self.kf.x[3:6] += acc * self.dt
        # Bias is modelled as random walk → no change in mean, only covariance.

        self.kf.P = self.kf.F @ self.kf.P @ self.kf.F.T + self.Q

    def update_stationary(self, lin_acc_w: np.ndarray) -> None:
        """Apply a zero-velocity / zero-acceleration correction."""

        self.kf.update(np.zeros(3), R=self.R_vel, H=self.H_vel)
        self.kf.update(np.asarray(lin_acc_w, dtype=np.float64), R=self.R_bias, H=self.H_bias)

    @property
    def position(self) -> np.ndarray:
        return self.kf.x[:3, 0].astype(np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[3:6, 0].astype(np.float32)

    @property
    def accel_bias(self) -> np.ndarray:
        return self.kf.x[6:9, 0].astype(np.float32)
