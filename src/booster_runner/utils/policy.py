import numpy as np
import onnxruntime
import torch

from .motion import MotionPlayer
from .rotate import (
    quat_conjugate,
    quat_multiply,
    quat_rotate_vector,
    quat_to_rotation_matrix,
    rotation_matrix_to_6d,
)


class Policy:
    def __init__(self, cfg, motion_file_path):
        self.cfg = cfg
        policy_path = self.cfg["policy"]["policy_path"]

        # Detect model type from file extension
        if policy_path.endswith('.onnx'):
            self.model_type = 'onnx'
            try:
                self.onnx_session = onnxruntime.InferenceSession(policy_path)
                print(f"Loaded ONNX model from {policy_path}")
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")
                raise
        elif policy_path.endswith('.pt'):
            self.model_type = 'torchscript'
            try:
                self.policy = torch.jit.load(policy_path)
                self.policy.eval()
                print(f"Loaded TorchScript model from {policy_path}")
            except Exception as e:
                print(f"Failed to load TorchScript model: {e}")
                raise
        else:
            raise ValueError(f"Unknown model type for {policy_path}. Expected .onnx or .pt")

        # Initialize motion player (required for both model types)
        anchor_body_index = self.cfg["policy"]["anchor_body_index"]
        self.motion_player = MotionPlayer(
            motion_path=motion_file_path,
            anchor_body_index=anchor_body_index,
        )

        print(f"Motion player initialized:")
        print(f"  - Num joints: {self.motion_player.num_joints}")
        print(f"  - Command dim: {self.motion_player.command_dim}")

        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(
            self.cfg["common"]["default_qpos"], dtype=np.float32
        )
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = (
            self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        )

        # Motion anchor tracking variables
        self.motion_anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.motion_anchor_pos = np.zeros(3, dtype=np.float32)

        # ONNX-specific output storage (for future use)
        if self.model_type == 'onnx':
            self.last_joint_pos = np.zeros(22, dtype=np.float32)
            self.last_joint_vel = np.zeros(22, dtype=np.float32)
            self.last_body_pos_w = np.zeros((25, 3), dtype=np.float32)
            self.last_body_quat_w = np.zeros((25, 4), dtype=np.float32)
            self.last_body_lin_vel_w = np.zeros((25, 3), dtype=np.float32)
            self.last_body_ang_vel_w = np.zeros((25, 3), dtype=np.float32)

    def inference(
        self,
        time_now: float,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        base_quat: np.ndarray,
    ) -> np.ndarray:
        """Run policy inference for motion tracking.

        Args:
            time_now: Current time
            dof_pos: Joint positions (22 dims)
            dof_vel: Joint velocities (22 dims)
            base_lin_vel: Base linear velocity (3 dims)
            base_ang_vel: Base angular velocity (3 dims)
            base_quat: Base orientation quaternion (4 dims, [w, x, y, z])

        Returns:
            Target joint positions (22 dims)
        """
        # Get motion command and anchor quaternion from MotionPlayer
        motion_command, self.motion_anchor_quat = self.motion_player.step(
            self.policy_interval
        )

        # Build observation vector (125 dims)
        # 0:44 - command (22 joint positions + 22 joint velocities)
        self.obs[0:44] = motion_command

        # 44:47 - motion_anchor_pos_b (motion anchor position in robot body frame)
        # For real robot deployment, we assume the motion anchor position is at the
        # same location as the robot anchor (Trunk), so relative position is [0, 0, 0]
        # This simplifies the implementation for hardware deployment
        self.obs[44:47] = 0.0

        # 47:53 - motion_anchor_ori_b (motion anchor orientation in robot body frame)
        # Compute relative orientation between motion anchor and robot base
        relative_quat = quat_multiply(
            quat_conjugate(base_quat), self.motion_anchor_quat
        )
        relative_rot_mat = quat_to_rotation_matrix(relative_quat)
        self.obs[47:53] = rotation_matrix_to_6d(relative_rot_mat)

        # 53:56 - base_lin_vel
        self.obs[53:56] = base_lin_vel * self.cfg["policy"]["normalization"]["lin_vel"]

        # 56:59 - base_ang_vel
        self.obs[56:59] = (
            base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        )

        # 59:81 - joint_pos (relative to default positions, all 22 joints)
        self.obs[59:81] = (
            dof_pos - self.default_dof_pos
        ) * self.cfg["policy"]["normalization"]["dof_pos"]

        # 81:103 - joint_vel (all 22 joints)
        self.obs[81:103] = dof_vel * self.cfg["policy"]["normalization"]["dof_vel"]

        # 103:125 - actions (previous actions, all 22 dims)
        self.obs[103:125] = self.actions

        # Run policy inference
        if self.model_type == 'torchscript':
            # TorchScript inference
            self.actions[:] = (
                self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()
            )
        else:  # onnx
            # ONNX inference
            onnx_inputs = {
                'obs': self.obs.reshape(1, -1).astype(np.float32),
                'time_step': np.array([[self.motion_player._frame_idx]], dtype=np.float32)
            }

            results = self.onnx_session.run(
                output_names=['actions', 'joint_pos', 'joint_vel', 'body_pos_w',
                              'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'],
                input_feed=onnx_inputs
            )

            # Extract actions (first output)
            self.actions[:] = results[0].flatten()

            # Store other outputs for future use
            self.last_joint_pos[:] = results[1].flatten()
            self.last_joint_vel[:] = results[2].flatten()
            self.last_body_pos_w[:] = results[3].squeeze(0)
            self.last_body_quat_w[:] = results[4].squeeze(0)
            self.last_body_lin_vel_w[:] = results[5].squeeze(0)
            self.last_body_ang_vel_w[:] = results[6].squeeze(0)

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

    def get_onnx_outputs(self):
        """Return last ONNX inference outputs for debugging/logging.

        Returns:
            dict or None: Dictionary containing ONNX outputs if using ONNX model,
                         None if using TorchScript model.
        """
        if self.model_type != 'onnx':
            return None
        return {
            'joint_pos': self.last_joint_pos.copy(),
            'joint_vel': self.last_joint_vel.copy(),
            'body_pos_w': self.last_body_pos_w.copy(),
            'body_quat_w': self.last_body_quat_w.copy(),
            'body_lin_vel_w': self.last_body_lin_vel_w.copy(),
            'body_ang_vel_w': self.last_body_ang_vel_w.copy(),
        }
