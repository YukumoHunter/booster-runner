from __future__ import annotations

import numpy as np
import onnxruntime
import torch

from .reference_observation import ReferenceObservationBuilder
from .rotate import (
    quat_from_yaw,
    quat_to_rotation_matrix,
    quat_yaw,
    rotation_matrix_to_quat,
)


class Policy:
    def __init__(self, cfg, playback_fps=None):
        self.cfg = cfg
        self.playback_fps = playback_fps
        policy_path = self.cfg["policy"]["policy_path"]

        # Detect model type from file extension
        if policy_path.endswith(".onnx"):
            self.model_type = "onnx"
            try:
                self.onnx_session = onnxruntime.InferenceSession(policy_path)
                print(f"Loaded ONNX model from {policy_path}")
                # Extract metadata from ONNX model
                self._extract_onnx_metadata()
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")
                raise
        elif policy_path.endswith(".pt"):
            self.model_type = "torchscript"
            try:
                self.policy = torch.jit.load(policy_path)
                self.policy.eval()
                print(f"Loaded TorchScript model from {policy_path}")
                # For TorchScript, use default values
                self.anchor_body_name = "trunk"
                self.anchor_body_index = 0
                self.body_names = []
            except Exception as e:
                print(f"Failed to load TorchScript model: {e}")
                raise
        else:
            raise ValueError(
                f"Unknown model type for {policy_path}. Expected .onnx or .pt"
            )

        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _extract_onnx_metadata(self):
        """Extract metadata from ONNX model."""
        metadata = self.onnx_session.get_modelmeta().custom_metadata_map

        # Extract anchor body name
        self.anchor_body_name = metadata.get("anchor_body_name", "trunk")

        # Extract body names (comma-separated string)
        body_names_str = metadata.get("body_names", "")
        self.body_names = [
            name.strip() for name in body_names_str.split(",") if name.strip()
        ]

        # Find anchor body index in body_names list
        if self.anchor_body_name in self.body_names:
            self.anchor_body_index = self.body_names.index(self.anchor_body_name)
        else:
            self.anchor_body_index = 0  # Fallback to trunk

        print(f"ONNX Metadata:")
        print(
            f"  - Anchor body: {self.anchor_body_name} (index {self.anchor_body_index})"
        )
        print(f"  - Body names: {self.body_names}")

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(
            self.cfg["common"]["default_qpos"], dtype=np.float32
        )
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.default_dof_vel = np.zeros_like(self.default_dof_pos)
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.raw_actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)

        # Add timestep tracking for ONNX motion decoding
        self.time_step = 0

        # Use custom playback FPS if provided, otherwise use policy decimation
        if self.playback_fps is not None:
            self.policy_interval = 1.0 / self.playback_fps
            print(
                f"Using custom playback interval: {self.policy_interval:.4f}s ({self.playback_fps} FPS)"
            )
        else:
            self.policy_interval = (
                self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
            )

        # Motion anchor tracking variables
        self.motion_anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.motion_anchor_pos = np.zeros(3, dtype=np.float32)

        self.obs_builder = ReferenceObservationBuilder(
            default_joint_pos=self.default_dof_pos,
            default_joint_vel=self.default_dof_vel,
            obs_dim=self.cfg["policy"]["num_observations"],
        )
        self.motion_anchor_pos: np.ndarray | None = None
        self.motion_anchor_quat: np.ndarray | None = None
        self._reset_motion_alignment()

    def _reset_motion_alignment(self):
        self._alignment_ready = False
        self._world_to_init_rot = np.eye(3, dtype=np.float32)
        self._world_to_init_trans = np.zeros(3, dtype=np.float32)

    def _maybe_initialize_motion_alignment(
        self, robot_anchor_pos_w: np.ndarray, robot_anchor_quat_w: np.ndarray
    ) -> None:
        if self._alignment_ready:
            return
        if self.motion_anchor_pos is None or self.motion_anchor_quat is None:
            return

        robot_yaw = quat_yaw(robot_anchor_quat_w)
        motion_yaw = quat_yaw(self.motion_anchor_quat)

        R_world_anchor = quat_to_rotation_matrix(quat_from_yaw(robot_yaw))
        R_init_anchor = quat_to_rotation_matrix(quat_from_yaw(motion_yaw))

        self._world_to_init_rot = (R_world_anchor @ R_init_anchor.T).astype(np.float32)
        self._world_to_init_trans = (
            robot_anchor_pos_w - self._world_to_init_rot @ self.motion_anchor_pos
        ).astype(np.float32)
        self._alignment_ready = True

    def _transform_motion_position(self, pos_w: np.ndarray) -> np.ndarray:
        if not self._alignment_ready:
            return pos_w
        return (self._world_to_init_rot @ pos_w) + self._world_to_init_trans

    def _transform_motion_orientation(self, quat_w: np.ndarray) -> np.ndarray:
        if not self._alignment_ready:
            return quat_w
        rot = quat_to_rotation_matrix(quat_w)
        aligned_rot = self._world_to_init_rot @ rot
        return rotation_matrix_to_quat(aligned_rot)

    def inference(
        self,
        time_now: float,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        base_quat: np.ndarray,
        base_pos_w: np.ndarray,
    ) -> np.ndarray:
        """Run policy inference for motion tracking.

        Args:
            time_now: Current time
            dof_pos: Joint positions (22 dims)
            dof_vel: Joint velocities (22 dims)
            base_lin_vel: Base linear velocity (3 dims)
            base_ang_vel: Base angular velocity (3 dims)
            base_quat: Base orientation quaternion (4 dims, [w, x, y, z])
            base_pos_w: Base position in world frame (3 dims)

        Returns:
            Target joint positions (22 dims)
        """
        # For ONNX models, extract reference motion from model outputs
        # For TorchScript, we'll handle it separately (if needed)
        if self.model_type == "onnx":
            # First run ONNX to get reference motion
            onnx_inputs_ref = {
                "obs": self.obs.reshape(1, -1).astype(np.float32),
                "time_step": np.array([[self.time_step]], dtype=np.float32),
            }

            # Request all motion outputs
            output_names = [
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            ]
            results = self.onnx_session.run(
                output_names=output_names,
                input_feed=onnx_inputs_ref,
            )

            # Extract outputs
            actions_out, joint_pos_out, joint_vel_out, body_pos_out, body_quat_out = (
                results
            )

            # Debug: print shapes
            print(f"ONNX output shapes:")
            print(f"  actions: {actions_out.shape}")
            print(f"  joint_pos: {joint_pos_out.shape}")
            print(f"  joint_vel: {joint_vel_out.shape}")
            print(f"  body_pos_w: {body_pos_out.shape}")
            print(f"  body_quat_w: {body_quat_out.shape}")
            print(f"  anchor_body_index: {self.anchor_body_index}")

            # Store action outputs
            self.ref_actions = actions_out.flatten()

            # Store reference motion outputs
            self.ref_joint_pos = joint_pos_out.flatten()
            self.ref_joint_vel = joint_vel_out.flatten()
            # Remove batch dimension first: (1, 25, 3) -> (25, 3)
            self.ref_body_pos_w = body_pos_out[0]
            self.ref_body_quat_w = body_quat_out[0]

            # Extract anchor body pose (just one body from all bodies)
            # Now indexing (25, 3) with [anchor_body_index] gives (3,)
            self.motion_anchor_pos = self.ref_body_pos_w[self.anchor_body_index].copy()
            self.motion_anchor_quat = self.ref_body_quat_w[
                self.anchor_body_index
            ].copy()

            print(
                f"  motion_anchor_pos shape after indexing: {self.motion_anchor_pos.shape}"
            )
            print(
                f"  motion_anchor_quat shape after indexing: {self.motion_anchor_quat.shape}"
            )

            # Build command from ONNX reference motion (not from NPZ!)
            motion_command = np.concatenate(
                [self.ref_joint_pos, self.ref_joint_vel], axis=0
            )

            # Increment timestep for next inference
            self.time_step += 1
        else:
            # For TorchScript, we need to handle this differently
            # For now, use zeros (this path needs proper implementation if TorchScript is used)
            motion_command = np.zeros(
                self.cfg["policy"]["num_actions"] * 2, dtype=np.float32
            )
            self.motion_anchor_pos = np.zeros(3, dtype=np.float32)
            self.motion_anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Align the first motion frame with the measured robot pose
        self._maybe_initialize_motion_alignment(
            robot_anchor_pos_w=base_pos_w, robot_anchor_quat_w=base_quat
        )

        aligned_anchor_pos = self._transform_motion_position(self.motion_anchor_pos)
        aligned_anchor_quat = self._transform_motion_orientation(
            self.motion_anchor_quat
        )

        # Assemble observation via shared builder to match simulation exactly
        self.obs[:] = self.obs_builder.build(
            command=motion_command,
            motion_anchor_pos_w=aligned_anchor_pos,
            motion_anchor_quat_w=aligned_anchor_quat,
            robot_anchor_pos_w=base_pos_w,
            robot_anchor_quat_w=base_quat,
            base_lin_vel_b=base_lin_vel,
            base_ang_vel_b=base_ang_vel,
            joint_pos=dof_pos,
            joint_vel=dof_vel,
            last_action_raw=self.raw_actions,
        )

        # Run policy inference
        if self.model_type == "torchscript":
            # TorchScript inference
            self.actions[:] = (
                self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()
            )
            self.raw_actions[:] = self.actions.copy()
        else:  # onnx
            # For ONNX, we already extracted actions when getting reference motion above
            # So we just use those actions
            self.actions[:] = self.ref_actions
            self.raw_actions[:] = self.ref_actions.copy()

        # Clip actions
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )

        # Compute target joint positions (all 22 joints)
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[:] += (
            np.array(self.cfg["policy"]["control"]["action_scales"]) * self.actions
        )

        return self.dof_targets

    def get_reference_motion(self) -> np.ndarray:
        """Get reference motion directly from ONNX without running policy.

        Returns:
            Target joint positions from ONNX encoded motion (22 dims)
        """
        if self.model_type == "onnx":
            # Run ONNX to get reference motion only
            onnx_inputs = {
                "obs": np.zeros(
                    (1, self.cfg["policy"]["num_observations"]), dtype=np.float32
                ),
                "time_step": np.array([[self.time_step]], dtype=np.float32),
            }

            results = self.onnx_session.run(
                output_names=["joint_pos"],
                input_feed=onnx_inputs,
            )

            self.time_step += 1
            return results[0].flatten()
        else:
            # For TorchScript, return zeros (needs proper implementation)
            return np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)

    def get_first_frame_motion(self) -> np.ndarray:
        """Get the first frame of reference motion from ONNX.

        Returns:
            Joint positions from frame 0 (22 dims)
        """
        if self.model_type == "onnx":
            # Reset timestep to 0 and get first frame
            saved_timestep = self.time_step
            self.time_step = 0

            onnx_inputs = {
                "obs": np.zeros(
                    (1, self.cfg["policy"]["num_observations"]), dtype=np.float32
                ),
                "time_step": np.array([[0]], dtype=np.float32),
            }

            results = self.onnx_session.run(
                output_names=["joint_pos"],
                input_feed=onnx_inputs,
            )

            # Restore timestep (don't increment)
            self.time_step = saved_timestep
            return results[0].flatten()
        else:
            # For TorchScript, return default positions
            return self.default_dof_pos.copy()

    def reset(self):
        """Reset policy state and timestep counter."""
        self.time_step = 0
        self.raw_actions[:] = 0
        self.actions[:] = 0
        self.motion_anchor_pos = None
        self.motion_anchor_quat = None
        self._reset_motion_alignment()
