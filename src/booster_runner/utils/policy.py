import numpy as np
import torch

import onnxruntime


class Policy:
    def __init__(self, cfg):
        policy_path = cfg["policy"]["policy_path"]
        assert policy_path.endswith(".onnx")

        self.cfg = cfg
        self.inference_session = onnxruntime.InferenceSession(policy_path)

        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(
            self.cfg["common"]["default_qpos"], dtype=np.float32
        )
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = (
            self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        )

    # +------------------------------------------------------------+
    # | Active Observation Terms in Group: 'policy' (shape: (75,)) |
    # +-----------+----------------------------------+-------------+
    # |   Index   | Name                             |    Shape    |
    # +-----------+----------------------------------+-------------+
    # |     0     | base_ang_vel                     |     (3,)    |
    # |     1     | projected_gravity                |     (3,)    |
    # |     2     | joint_pos                        |    (22,)    |
    # |     3     | joint_vel                        |    (22,)    |
    # |     4     | actions                          |    (22,)    |
    # |     5     | command                          |     (3,)    |
    # +-----------+----------------------------------+-------------+
    def inference(
        self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw
    ):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(
            self.commands - self.smoothed_commands, *clip_range
        )

        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = 0.0
        else:
            self.gait_frequency = self.cfg["policy"]["gait_frequency"]

        self.obs[0:3] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[3:6] = (
            projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        )
        self.obs[6:28] = (dof_pos - self.default_dof_pos) * self.cfg["policy"][
            "normalization"
        ]["dof_pos"]
        self.obs[28:50] = dof_vel[11:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[50:72] = self.actions
        self.obs[72] = (
            self.smoothed_commands[0]
            * self.cfg["policy"]["normalization"]["lin_vel"]
            * (self.gait_frequency > 1.0e-8)
        )
        self.obs[73] = (
            self.smoothed_commands[1]
            * self.cfg["policy"]["normalization"]["lin_vel"]
            * (self.gait_frequency > 1.0e-8)
        )
        self.obs[74] = (
            self.smoothed_commands[2]
            * self.cfg["policy"]["normalization"]["ang_vel"]
            * (self.gait_frequency > 1.0e-8)
        )

        onnx_inputs = {
            "obs": self.obs.reshape(1, -1).astype(np.float32),
        }

        self.actions[:] = self.onnx_session.run(
            output_names=[
                "joint_pos",
            ],
            input_feed=onnx_inputs,
        )[0]

        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets += (
            np.array(self.cfg["policy"]["control"]["action_scales"]) * self.actions
        )

        return self.dof_targets
