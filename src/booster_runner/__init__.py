import numpy as np
import time
import yaml
import logging
import threading

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
from .utils.orientation_filter import OrientationFilter
from .utils.policy import Policy
from .utils.remote_control_service import RemoteControlService
from .utils.rerun_logger import OrientationRerunLogger
from .utils.rotate import quat_conjugate, quat_rotate_vector
from .utils.timer import TimerConfig, Timer


class Controller:
    def __init__(self, cfg_file, motion_file) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Initialize components
        self.remoteControlService = RemoteControlService()
        self.policy = Policy(cfg=self.cfg, motion_file_path=motion_file)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()

    def _init_timer(self):
        dt = self.cfg["common"]["dt"]
        self.timer = Timer(TimerConfig(time_step=dt))
        self.dt = dt
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(22, dtype=np.float32)
        self.dof_vel = np.zeros(22, dtype=np.float32)

        self.dof_target = np.zeros(22, dtype=np.float32)
        self.filtered_dof_target = np.zeros(22, dtype=np.float32)
        self.dof_pos_latest = np.zeros(22, dtype=np.float32)
        self.gravity_magnitude = 9.81
        self.world_gravity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.orientation_filter = OrientationFilter(
            dt=self.cfg["common"]["dt"],
            gravity_magnitude=self.gravity_magnitude,
        )
        self.rerun_logger = OrientationRerunLogger(stream_path="robot/base_pose")
        self._imu_acc_attributes = (
            "acc",
            "accel",
            "acceleration",
            "acc_3d",
            "linear_acceleration",
        )
        self._imu_vel_attributes = ("velocity", "lin_vel", "linear_velocity")
        self.last_imu_update_time = self.timer.get_time()
        self.orientation_log_period = 0.5
        self.next_orientation_log_time = self.timer.get_time()

    def _extract_imu_vector(self, imu_state, attr_names):
        for attr in attr_names:
            if not hasattr(imu_state, attr):
                continue
            value = getattr(imu_state, attr)
            if value is None:
                continue
            vector = np.asarray(value, dtype=np.float32).flatten()
            if vector.size >= 3:
                return vector[:3].copy()
        return None

    def _integrate_base_velocity(self, accel_body: np.ndarray, dt: float) -> None:
        if not self.orientation_filter.initialized:
            return
        accel_body = np.asarray(accel_body, dtype=np.float32)
        gravity_body = self.projected_gravity * self.gravity_magnitude
        linear_acc_body = accel_body - gravity_body
        linear_acc_world = quat_rotate_vector(self.base_quat, linear_acc_body)
        self.base_lin_vel[:] += linear_acc_world * dt

    def _log_orientation(self, time_now: float) -> None:
        self.rerun_logger.log_quaternion(self.base_quat)

        if time_now < self.next_orientation_log_time:
            return
        self.next_orientation_log_time = time_now + self.orientation_log_period

        roll, pitch, yaw = self._quat_to_rpy(self.base_quat)
        self.logger.info(
            "Tracked orientation | quat=%s | rpy_deg=(%.2f, %.2f, %.2f)",
            np.array2string(self.base_quat, precision=3),
            np.degrees(roll),
            np.degrees(pitch),
            np.degrees(yaw),
        )

    @staticmethod
    def _quat_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
        w, x, y, z = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

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
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
            if time_now >= self.next_inference_time:
                rpy = low_state_msg.imu_state.rpy

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

            gyro = np.asarray(low_state_msg.imu_state.gyro, dtype=np.float32)
            accel = self._extract_imu_vector(
                low_state_msg.imu_state, self._imu_acc_attributes
            )
            if not self.orientation_filter.initialized:
                self.orientation_filter.reset(rpy_to_quat(rpy[0], rpy[1], rpy[2]))

            elapsed = max(self.dt, time_now - self.last_imu_update_time)
            self.last_imu_update_time = time_now

            self.base_ang_vel[:] = gyro
            self.base_quat[:] = self.orientation_filter.update(
                gyro=gyro, accel=accel, dt=elapsed
            )
            self.projected_gravity[:] = quat_rotate_vector(
                quat_conjugate(self.base_quat), self.world_gravity_dir
            )
            self._log_orientation(time_now)

            velocity = self._extract_imu_vector(
                low_state_msg.imu_state, self._imu_vel_attributes
            )
            if velocity is not None:
                self.base_lin_vel[:] = velocity
            elif accel is not None:
                self._integrate_base_velocity(accel, elapsed)
            else:
                self.base_lin_vel.fill(0.0)

            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

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
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()

        self.dof_target[:] = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_lin_vel=self.base_lin_vel,
            base_ang_vel=self.base_ang_vel,
            base_quat=self.base_quat,
        )

        inference_time = time.perf_counter()
        self.logger.debug(
            f"Inference took {(inference_time - start_time) * 1000:.4f} ms"
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
        "--motion",
        required=True,
        type=str,
        help="Path to motion file (.npz) for tracking",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="127.0.0.1",
        help="Network interface for SDK communication.",
    )
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)

    print(f"Starting custom controller, connecting to {args.net} ...")
    print(f"Loading motion file: {args.motion}")
    ChannelFactory.Instance().Init(0, args.net)

    try:
        with Controller(cfg_file, args.motion) as controller:
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
