import numpy as np
import time
import yaml
import logging
import threading
from dataclasses import dataclass

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
)

from .utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from .utils.remote_control_service import RemoteControlService
from .utils.pose_filter import BasePoseFilter
from .utils.rotate import quat_conjugate, quat_rotate_vector, rotate_vector_inverse_rpy
from .utils.timer import TimerConfig, Timer
from .utils.policy import Policy


@dataclass
class MotorStateBuffer:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau_est: np.ndarray


@dataclass
class ImuStateBuffer:
    rpy: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray


@dataclass
class LowStateBuffers:
    imu: ImuStateBuffer
    motor_state_serial: MotorStateBuffer
    motor_state_parallel: MotorStateBuffer


class Controller:
    def __init__(self, cfg_file, playback_only=False, playback_fps=None) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Initialize components
        self.remoteControlService = RemoteControlService()
        self.policy = Policy(cfg=self.cfg, playback_fps=playback_fps)
        self.playback_only = playback_only
        self.playback_fps = playback_fps

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()
        self.in_initialization = False
        self.init_start_time = None

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self._base_lin_vel_w = np.zeros(3, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.base_pos_w = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self._gravity_w = np.array([0.0, 0.0, 9.81], dtype=np.float32)
        self.dof_pos = np.zeros(22, dtype=np.float32)
        self.dof_vel = np.zeros(22, dtype=np.float32)

        self.dof_target = np.zeros(22, dtype=np.float32)
        self.filtered_dof_target = np.zeros(22, dtype=np.float32)
        self.dof_pos_latest = np.zeros(22, dtype=np.float32)
        self._last_state_time = None
        self.low_state_buffers: LowStateBuffers | None = None
        self.pose_filter = BasePoseFilter(dt=self.cfg["common"]["dt"])

    def _create_motor_state_buffer(self, length: int) -> MotorStateBuffer:
        return MotorStateBuffer(
            q=np.zeros(length, dtype=np.float32),
            dq=np.zeros(length, dtype=np.float32),
            ddq=np.zeros(length, dtype=np.float32),
            tau_est=np.zeros(length, dtype=np.float32),
        )

    def _ensure_low_state_buffers(self, low_state_msg: LowState) -> LowStateBuffers:
        serial_count = len(low_state_msg.motor_state_serial)
        parallel_count = len(low_state_msg.motor_state_parallel)

        if self.low_state_buffers is not None:
            serial_mismatch = (
                self.low_state_buffers.motor_state_serial.q.shape[0] != serial_count
            )
            parallel_mismatch = (
                self.low_state_buffers.motor_state_parallel.q.shape[0] != parallel_count
            )
            if serial_mismatch or parallel_mismatch:
                self.low_state_buffers = None

        if self.low_state_buffers is None:
            self.low_state_buffers = LowStateBuffers(
                imu=ImuStateBuffer(
                    rpy=np.zeros(3, dtype=np.float32),
                    gyro=np.zeros(3, dtype=np.float32),
                    acc=np.zeros(3, dtype=np.float32),
                ),
                motor_state_serial=self._create_motor_state_buffer(serial_count),
                motor_state_parallel=self._create_motor_state_buffer(parallel_count),
            )
        return self.low_state_buffers

    def get_low_state_buffers(self) -> LowStateBuffers | None:
        """Expose the cached low state buffers for diagnostics or logging."""
        return self.low_state_buffers

    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if (
            abs(low_state_msg.imu_state.rpy[0]) > 1.0
            or abs(low_state_msg.imu_state.rpy[1]) > 1.0
        ):
            self.logger.warning(
                "IMU base rpy values are too large: {}".format(
                    low_state_msg.imu_state.rpy
                )
            )
            self.running = False
        self.running = True
        buffers = self._ensure_low_state_buffers(low_state_msg)

        # Cache IMU feedback
        buffers.imu.rpy[:] = np.array(low_state_msg.imu_state.rpy, dtype=np.float32)
        buffers.imu.gyro[:] = np.array(low_state_msg.imu_state.gyro, dtype=np.float32)
        buffers.imu.acc[:] = np.array(low_state_msg.imu_state.acc, dtype=np.float32)

        # Cache motor state feedback
        serial_buffer = buffers.motor_state_serial
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            serial_buffer.q[i] = motor.q
            serial_buffer.dq[i] = motor.dq
            serial_buffer.ddq[i] = motor.ddq
            serial_buffer.tau_est[i] = motor.tau_est

        parallel_buffer = buffers.motor_state_parallel
        for i, motor in enumerate(low_state_msg.motor_state_parallel):
            parallel_buffer.q[i] = motor.q
            parallel_buffer.dq[i] = motor.dq
            parallel_buffer.ddq[i] = motor.ddq
            parallel_buffer.tau_est[i] = motor.tau_est

        joint_count = min(serial_buffer.q.shape[0], self.dof_pos_latest.shape[0])
        self.dof_pos_latest[:joint_count] = serial_buffer.q[:joint_count]

        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        if time_now >= self.next_inference_time:
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                buffers.imu.rpy[0],
                buffers.imu.rpy[1],
                buffers.imu.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = buffers.imu.gyro

            rpy = buffers.imu.rpy

            def rpy_to_quat(roll, pitch, yaw):
                cy = np.cos(yaw * 0.5)
                sy = np.sin(yaw * 0.5)
                cp = np.cos(pitch * 0.5)
                sp = np.sin(pitch * 0.5)
                cr = np.cos(roll * 0.5)
                sr = np.sin(roll * 0.5)

                w = cr * cp * cy + sr * sp * sy
                x = sr * cp * cy - cr * sp * sy
                y = cr * sp * cy + sr * cp * sy
                z = cr * cp * sy - sr * sp * cy
                return np.array([w, x, y, z], dtype=np.float32)

            quaternion = rpy_to_quat(rpy[0], rpy[1], rpy[2])

            self.base_quat[:] = quaternion
            if self._last_state_time is None:
                dt = self.cfg["common"]["dt"]
            else:
                dt = max(1e-4, time_now - self._last_state_time)
            self._last_state_time = time_now

            acc_b = buffers.imu.acc
            acc_w = quat_rotate_vector(self.base_quat, acc_b)
            lin_acc_w = acc_w - self._gravity_w
            self.pose_filter.predict(lin_acc_w)

            gyro_norm = np.linalg.norm(buffers.imu.gyro)
            acc_norm = np.linalg.norm(lin_acc_w)
            if gyro_norm < 0.15 and acc_norm < 0.25:
                self.pose_filter.update_stationary(lin_acc_w)

            self._base_lin_vel_w[:] = self.pose_filter.velocity
            vel_norm = np.linalg.norm(self._base_lin_vel_w)
            if vel_norm > 5.0:
                self._base_lin_vel_w *= 5.0 / vel_norm

            self.base_pos_w[:] = self.pose_filter.position
            self.base_lin_vel[:] = quat_rotate_vector(
                quat_conjugate(self.base_quat), self._base_lin_vel_w
            )

            self.dof_pos[:joint_count] = serial_buffer.q[:joint_count]
            self.dof_vel[:joint_count] = serial_buffer.dq[:joint_count]

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.remoteControlService.close()
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if (
            hasattr(self, "publish_runner")
            and getattr(self, "publish_runner") is not None
        ):
            self.publish_runner.join(timeout=1.0)

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(22):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        self.logger.debug(f"Send cmd took {(send_time - start_time) * 1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        end_time = time.perf_counter()
        self.logger.debug(f"Change mode took {(end_time - send_time) * 1000:.4f} ms")

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)

        self.policy.reset()
        # Get first frame of reference motion
        first_frame_motion = self.policy.get_first_frame_motion()

        # Set command to first frame positions
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        for i in range(22):
            self.low_cmd.motor_cmd[i].q = first_frame_motion[i]

        self._send_cmd(self.low_cmd)

        # Initialize timing with 3-second delay for initialization phase
        init_time = self.timer.get_time()
        self.init_start_time = init_time
        self.in_initialization = True
        self.next_inference_time = init_time + 3.0  # Delay inference by 3 seconds
        self.next_publish_time = init_time

        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        time_now = self.timer.get_time()

        # Handle initialization phase
        if self.in_initialization:
            if time_now < self.next_inference_time:
                # Still in initialization - keep holding frame 0
                # The publish thread continues sending the frame 0 command
                time.sleep(0.001)
                return
            else:
                # Initialization complete - transition to normal operation
                self.in_initialization = False
                self.logger.info("Initialization complete - starting policy inference")
                # next_inference_time is already set correctly for first inference

        # Continue with existing control logic
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()

        if self.playback_only:
            # Playback mode: use reference motion directly without inference
            self.dof_target[:] = self.policy.get_reference_motion()
        else:
            # Normal mode: run policy inference
            self.dof_target[:] = self.policy.inference(
                time_now=time_now,
                dof_pos=self.dof_pos,
                dof_vel=self.dof_vel,
                base_lin_vel=self.base_lin_vel,
                base_ang_vel=self.base_ang_vel,
                base_quat=self.base_quat,
                base_pos_w=self.base_pos_w,
            )

        inference_time = time.perf_counter()
        self.logger.debug(
            f"{'Playback' if self.playback_only else 'Inference'} took {(inference_time - start_time) * 1000:.4f} ms"
        )
        time.sleep(0.001)

    def _publish_cmd(self):
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.cfg["common"]["dt"]
            self.logger.debug(f"Next publish time: {self.next_publish_time}")

            self.filtered_dof_target = (
                self.filtered_dof_target * 0.8 + self.dof_target * 0.2
            )

            motor_cmd = self.low_cmd.motor_cmd
            for i in range(22):
                motor_cmd[i].q = self.filtered_dof_target[i]

            # Use series-parallel conversion for torque to avoid non-linearity
            for i in self.cfg["mech"]["parallel_mech_indexes"]:
                motor_cmd[i].q = self.dof_pos_latest[i]
                motor_cmd[i].tau = np.clip(
                    (self.filtered_dof_target[i] - self.dof_pos_latest[i])
                    * self.cfg["common"]["stiffness"][i],
                    -self.cfg["common"]["torque_limit"][i],
                    self.cfg["common"]["torque_limit"][i],
                )
                motor_cmd[i].kp = 0.0

            start_time = time.perf_counter()
            self._send_cmd(self.low_cmd)
            publish_time = time.perf_counter()
            self.logger.debug(
                f"Publish took {(publish_time - start_time) * 1000:.4f} ms"
            )
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


def main():
    import argparse
    import signal
    import sys
    import os

    controller = None

    def signal_handler(_sig, _frame):
        print("\nShutting down...")
        if controller:
            controller.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, type=str, help="Name of the configuration file."
    )
    parser.add_argument(
        "--net",
        type=str,
        default="127.0.0.1",
        help="Network interface for SDK communication.",
    )
    parser.add_argument(
        "--playback-only",
        action="store_true",
        help="Play back reference motion without running policy inference.",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=None,
        help="Playback frames per second (overrides motion file FPS and policy rate).",
    )
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)

    print(f"Starting custom controller, connecting to {args.net} ...")
    print(f"Motion will be loaded from ONNX model (encoded motion)")
    ChannelFactory.Instance().Init(0, args.net)

    if args.playback_only:
        print("Running in PLAYBACK-ONLY mode (no policy inference)")
    if args.playback_fps is not None:
        print(f"Using custom playback FPS: {args.playback_fps}")

    try:
        with Controller(
            cfg_file,
            playback_only=args.playback_only,
            playback_fps=args.playback_fps,
        ) as controller:
            time.sleep(2)  # Wait for channels to initialize
            print("Initialization complete.")
            controller.start_custom_mode_conditionally()
            controller.start_rl_gait_conditionally()

            try:
                while controller.running:
                    controller.run()
                controller.client.ChangeMode(RobotMode.kDamping)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Cleaning up...")
                controller.cleanup()
    except Exception as e:
        print(f"\nError occurred: {e}")
        if controller:
            controller.cleanup()
        raise
