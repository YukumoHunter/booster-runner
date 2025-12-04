"""Reference observation builder that mirrors Booster Lab policy inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .rotate import (
    quat_conjugate,
    quat_multiply,
    quat_rotate_vector,
    quat_to_rotation_matrix,
    rotation_matrix_to_6d,
)


@dataclass(frozen=True)
class ObservationTermLayout:
    """Metadata describing a contiguous observation slice."""

    name: str
    sl: slice
    description: str


def _relative_transform(
    parent_pos_w: np.ndarray,
    parent_quat_w: np.ndarray,
    child_pos_w: np.ndarray,
    child_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express child pose in the parent frame."""

    q_parent_inv = quat_conjugate(parent_quat_w)
    q_rel = quat_multiply(q_parent_inv, child_quat_w)
    pos_rel = quat_rotate_vector(q_parent_inv, child_pos_w - parent_pos_w)
    return pos_rel, q_rel


def _as_float_array(
    vec: np.ndarray | list[float], length: int, name: str
) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}")
    return arr


def _build_layout(num_joints: int, obs_dim: int) -> tuple[ObservationTermLayout, ...]:
    """Create deterministic slices for each policy observation term."""

    lengths: list[tuple[str, int, str]] = [
        (
            "command",
            num_joints * 2,
            "Reference motion joint positions + velocities (MotionCommand.command)",
        ),
        (
            "motion_anchor_pos_b",
            3,
            "Desired anchor translation expressed in the robot base frame",
        ),
        (
            "motion_anchor_ori_b",
            6,
            "First two columns of the relative rotation matrix (body -> motion anchor)",
        ),
        # ("base_lin_vel", 3, "IMU linear velocity measured in the base frame (m/s)"),
        ("base_ang_vel", 3, "IMU angular velocity measured in the base frame (rad/s)"),
        ("joint_pos", num_joints, "Joint positions relative to default pose"),
        ("joint_vel", num_joints, "Joint velocities relative to default (zero)"),
        ("actions", num_joints, "Previous raw action sent to the policy head"),
    ]

    start = 0
    layout: list[ObservationTermLayout] = []
    for name, length, desc in lengths:
        stop = start + length
        layout.append(
            ObservationTermLayout(name=name, sl=slice(start, stop), description=desc)
        )
        start = stop

    if start != obs_dim:
        raise ValueError(
            f"Observation layout adds up to {start} dims, expected {obs_dim}. "
            "Check joint count or update the layout."
        )
    return tuple(layout)


class ReferenceObservationBuilder:
    """Assemble the policy observation exactly like the simulator."""

    def __init__(
        self,
        default_joint_pos: np.ndarray,
        default_joint_vel: np.ndarray | None = None,
        obs_dim: int = 122,
    ) -> None:
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).copy()
        self.default_joint_vel = (
            np.zeros_like(self.default_joint_pos, dtype=np.float32)
            if default_joint_vel is None
            else np.asarray(default_joint_vel, dtype=np.float32).copy()
        )

        if self.default_joint_pos.ndim != 1:
            raise ValueError("default_joint_pos must be a 1-D array")
        if self.default_joint_vel.shape != self.default_joint_pos.shape:
            raise ValueError("default joint velocity must match position length")

        self.num_joints = self.default_joint_pos.shape[0]
        self.obs_dim = obs_dim

        self._layout = _build_layout(self.num_joints, obs_dim)
        self._slices: Mapping[str, slice] = {
            term.name: term.sl for term in self._layout
        }

    @property
    def layout(self) -> tuple[ObservationTermLayout, ...]:
        """Expose the observation layout metadata."""

        return self._layout

    def build(
        self,
        *,
        command: np.ndarray,
        motion_anchor_pos_w: np.ndarray,
        motion_anchor_quat_w: np.ndarray,
        robot_anchor_pos_w: np.ndarray,
        robot_anchor_quat_w: np.ndarray,
        base_lin_vel_b: np.ndarray,
        base_ang_vel_b: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_action_raw: np.ndarray,
    ) -> np.ndarray:
        """Create the 125-D policy observation vector.

        Args:
            command: Motion command buffer [joint_pos, joint_vel], shape (2 * num_joints,)
            motion_anchor_pos_w: Anchor position from motion (world frame), shape (3,)
            motion_anchor_quat_w: Anchor orientation from motion (world frame, wxyz)
            robot_anchor_pos_w: Robot trunk position in world frame, shape (3,)
            robot_anchor_quat_w: Robot trunk orientation in world frame (wxyz)
            base_lin_vel_b: IMU linear velocity expressed in base frame (3,)
            base_ang_vel_b: IMU angular velocity expressed in base frame (3,)
            joint_pos: Measured joint positions (num_joints,)
            joint_vel: Measured joint velocities (num_joints,)
            last_action_raw: Previous raw policy output after clipping (num_joints,)
        """

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        obs[self._slices["command"]] = _as_float_array(
            command, self.num_joints * 2, "command"
        )

        motion_anchor_pos_w = _as_float_array(
            motion_anchor_pos_w, 3, "motion_anchor_pos_w"
        )
        robot_anchor_pos_w = _as_float_array(
            robot_anchor_pos_w, 3, "robot_anchor_pos_w"
        )
        motion_anchor_quat_w = _as_float_array(
            motion_anchor_quat_w, 4, "motion_anchor_quat_w"
        )
        robot_anchor_quat_w = _as_float_array(
            robot_anchor_quat_w, 4, "robot_anchor_quat_w"
        )

        rel_pos_b, rel_quat_b = _relative_transform(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            motion_anchor_pos_w,
            motion_anchor_quat_w,
        )
        obs[self._slices["motion_anchor_pos_b"]] = rel_pos_b
        obs[self._slices["motion_anchor_ori_b"]] = rotation_matrix_to_6d(
            quat_to_rotation_matrix(rel_quat_b)
        )

        obs[self._slices["base_lin_vel"]] = _as_float_array(
            base_lin_vel_b, 3, "base_lin_vel_b"
        )
        obs[self._slices["base_ang_vel"]] = _as_float_array(
            base_ang_vel_b, 3, "base_ang_vel_b"
        )

        obs[self._slices["joint_pos"]] = (
            _as_float_array(joint_pos, self.num_joints, "joint_pos")
            - self.default_joint_pos
        )
        obs[self._slices["joint_vel"]] = (
            _as_float_array(joint_vel, self.num_joints, "joint_vel")
            - self.default_joint_vel
        )
        obs[self._slices["actions"]] = _as_float_array(
            last_action_raw, self.num_joints, "last_action_raw"
        )

        return obs
