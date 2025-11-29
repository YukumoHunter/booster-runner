import logging
import numpy as np

try:
    import rerun as rr  # type: ignore
except Exception:  # pragma: no cover
    rr = None


class OrientationRerunLogger:
    """Lazy-initialized helper that streams orientation quaternions to rerun."""

    def __init__(self, stream_path: str = "robot/base") -> None:
        self.stream_path = stream_path
        self._initialized = False
        self._available = rr is not None
        self._logger = logging.getLogger(__name__)

        if not self._available:
            self._logger.debug("rerun-sdk not available; orientation stream disabled.")

    def log_quaternion(self, quat: np.ndarray) -> None:
        if not self._available:
            return
        if not self._initialized:
            self._init_rerun()
        if not self._initialized:
            return

        quat = np.asarray(quat, dtype=np.float32)
        rotation = rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]])
        rr.log(self.stream_path, rr.Transform3D(rotation=rotation))

    def _init_rerun(self) -> None:
        if not self._available:
            return
        try:
            rr.init("booster-runner", spawn=False)
            rr.connect(addr="192.168.14.250:9876")
            self._initialized = True
        except Exception as exc:  # pragma: no cover
            self._available = False
            self._logger.warning("Failed to initialize rerun logging: %s", exc)
