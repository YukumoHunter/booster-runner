# Policy Observation Layout (Booster K1 Tracking)

## Source of truth
- Observation group defined in `booster_lab/tasks/tracking/tracking_env_cfg.py` (policy terms).
- Term implementations:
  - `MotionCommand.command` and anchor helpers: `mjlab/tasks/tracking/mdp/commands.py`, `mjlab/tasks/tracking/mdp/observations.py`.
  - Generic observation helpers: `mjlab/envs/mdp/observations.py`.
- Manager configuration ensures concatenation order is preserved, so slices below are authoritative.

## Term layout (125 dims)
| Indices (inclusive) | Dim | Name | Formula & Notes | Scaling / Noise |
| --- | --- | --- | --- | --- |
| 0–43 | 44 | `command` | `torch.cat([joint_pos, joint_vel], dim=1)` from the sampled `MotionCommand`. Values are the reference joint pose [rad] and velocity [rad/s] for the current motion frame. | No scaling, no noise. |
| 44–46 | 3 | `motion_anchor_pos_b` | `quat_inv(q_robot) * (p_motion - p_robot)` computed via `subtract_frame_transforms`. Expresses motion anchor translation in robot trunk frame. | Uniform noise per dim `U(-0.25, 0.25)` m. |
| 47–52 | 6 | `motion_anchor_ori_b` | `matrix_from_quat(quat_inv(q_robot) * q_motion)[..., :2].reshape(-1)` – first two columns of relative rotation matrix flattened. Matches sim 6D representation. | Uniform noise `U(-0.05, 0.05)` on each element. |
| 53–55 | 3 | `base_lin_vel` | Builtin sensor `robot/imu_lin_vel` (linear vel of IMU site expressed in body frame). | Uniform noise `U(-0.5, 0.5)` m/s. |
| 56–58 | 3 | `base_ang_vel` | Builtin sensor `robot/imu_ang_vel` (angular vel of IMU site in body frame). | Uniform noise `U(-0.2, 0.2)` rad/s. |
| 59–80 | 22 | `joint_pos` | `robot.joint_pos - robot.data.default_joint_pos`. Joint order matches config comment in `configs/K1.yaml`. | Uniform noise `U(-0.01, 0.01)` rad. |
| 81–102 | 22 | `joint_vel` | `robot.joint_vel - robot.data.default_joint_vel` (defaults are 0). | Uniform noise `U(-0.5, 0.5)` rad/s. |
| 103–124 | 22 | `actions` | `env.action_manager.action`, i.e., the previous raw policy action after Env-level clipping (see `mjlab/envs/mdp/observations.py`). For joint position actions this equals the normalized value in [-clip, clip], *not* the scaled joint target. | No scaling, no noise. |

## Reference builder
- The runtime observation assembly now lives in `src/booster_runner/utils/reference_observation.py` (`ReferenceObservationBuilder`).
- `Policy.inference()` uses this builder directly so the hardware path matches simulation bit-for-bit (once sensor frames are aligned).
- The builder exposes `layout` metadata that mirrors the table above for debugging/validation scripts.

## Sensor/frame requirements
- `motion_anchor_pos_b` & `motion_anchor_ori_b` require the robot trunk pose in the world frame. Provide `base_pos_w` and `base_quat` expressed in the same world frame used by the motion clips (origin at simulation world origin).
- IMU angular velocity must be in rad/s and in the trunk/body frame. Convert vendor units if needed.
- IMU linear velocity must be trunk-frame m/s. Rotate world-frame velocities through the base quaternion if that is what the SDK exposes.
- `actions` slice must store the unclipped raw action vector *before* applying per-joint stiffness scaling to produce joint targets. The builder expects the last applied policy output.

With these conventions, an offline A/B check comparing simulator and hardware snapshots should match up to floating-point noise.
