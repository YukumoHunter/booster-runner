from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class MotionPlayer:
    """Utility for stepping through motion capture trajectories exported from Booster Lab."""

    def __init__(
        self,
        motion_path: str,
        anchor_body_index: int = 0,
    ) -> None:
        path = Path(motion_path)
        if not path.exists():
            raise FileNotFoundError(f"Motion file '{motion_path}' does not exist.")

        data = np.load(path, allow_pickle=False)
        required_keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(
                f"Motion file '{motion_path}' is missing required keys: {missing}."
            )

        self._joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        self._joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)

        if self._joint_pos.shape != self._joint_vel.shape:
            raise ValueError(
                "joint_pos and joint_vel arrays must have identical shapes "
                f"(got {self._joint_pos.shape} and {self._joint_vel.shape})."
            )

        if body_pos.ndim != 3 or body_quat.ndim != 3:
            raise ValueError(
                "body_pos_w and body_quat_w arrays must have shape (frames, bodies, X). "
                f"Got {body_pos.shape} and {body_quat.shape}."
            )

        if anchor_body_index < 0 or anchor_body_index >= body_quat.shape[1]:
            raise IndexError(
                f"anchor_body_index {anchor_body_index} is outside valid range "
                f"[0, {body_quat.shape[1] - 1}]."
            )

        self._anchor_pos = body_pos[:, anchor_body_index]
        self._anchor_quat = body_quat[:, anchor_body_index]
        self._frame_count = self._joint_pos.shape[0]
        if self._frame_count == 0:
            raise ValueError("Motion file does not contain any frames.")

        fps = None
        if "fps" in data:
            fps_values = np.asarray(data["fps"], dtype=np.float32).flatten()
            if fps_values.size > 0 and fps_values[0] > 0:
                fps = float(fps_values[0])
        self._motion_dt = 1.0 / fps if fps is not None and fps > 0 else None
        self._frame_idx = 0
        self._time_accumulator = 0.0

    @property
    def num_joints(self) -> int:
        return self._joint_pos.shape[1]

    @property
    def command_dim(self) -> int:
        return self.num_joints * 2

    def step(self, dt: float | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the current command vector and advance the playback cursor."""
        command = np.concatenate(
            (self._joint_pos[self._frame_idx], self._joint_vel[self._frame_idx]),
            axis=0,
        ).astype(np.float32, copy=False)
        anchor_pos = self._anchor_pos[self._frame_idx].copy()
        anchor_quat = self._anchor_quat[self._frame_idx].copy()
        self._advance(dt)
        return command, anchor_pos, anchor_quat

    def _advance(self, dt: float | None) -> None:
        if self._motion_dt is None:
            self._frame_idx = (self._frame_idx + 1) % self._frame_count
            return

        step_dt = self._motion_dt if dt is None else max(dt, 0.0)
        self._time_accumulator += step_dt

        while self._time_accumulator >= self._motion_dt:
            self._time_accumulator -= self._motion_dt
            self._frame_idx = (self._frame_idx + 1) % self._frame_count
