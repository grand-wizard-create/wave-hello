"""Wave hello: your arm rises, waves hello, then folds back to rest. A gentle first app.

Runs on the SO-101 follower arm. Every move is streamed, one small step per tick, so no joint ever
jumps: glides go at most 45 degrees a second and the wave peaks just under that. The app never aims
at the folded rest pose itself. It glides back to where it started, held 10 degrees inside the
arm's limits, then ends, and the arm's own slow fold to rest does the last part.

The work runs as phases in ``loop()`` (``rise``, ``wave``, ``back``), so a Stop lands on any tick.
A camera is optional: when there is one, the app counts the frames it sees while it waves.
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Sequence
from typing import Any

from monorobot.sdk.app import App

log = logging.getLogger("wave-hello")

SO101 = frozenset({"hf_so101_follower", "sim_so101_follower"})
JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Where the arm says hello, in degrees.
HELLO_DEG: dict[str, float] = {
    "shoulder_pan": 0.0, "shoulder_lift": -40.0, "elbow_flex": 45.0, "wrist_flex": -15.0, "wrist_roll": 0.0,
}
# The wave, per joint: (amplitude in degrees, phase in periods) of a sine around HELLO_DEG.
# 15 degrees over PERIOD_S peaks at 43 degrees a second.
WAVE_DEG: dict[str, tuple[float, float]] = {"wrist_flex": (15.0, 0.0)}
PERIOD_S = 2.2
WAVES = 3
GRIPPER_OPEN = 0.6           # 0 closed, 1 open
GLIDE_DEG_PER_S = 45.0
CAMERA_EVERY_TICKS = 10

# Every target stays 10 degrees inside the follower's URDF limits, and the base turns at most 60
# degrees either way.
ENVELOPE_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-60.0, 60.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-86.8, 86.8),
    "wrist_flex": (-85.0, 85.0),
    "wrist_roll": (-147.0, 152.0),
}


def clip_to_envelope(joints_deg: Sequence[float]) -> list[float]:
    """The pose held inside ``ENVELOPE_DEG``, joint by joint."""
    return [min(max(float(v), ENVELOPE_DEG[name][0]), ENVELOPE_DEG[name][1]) for name, v in zip(JOINTS, joints_deg)]


def wave_pose(t: float) -> list[float]:
    """The arm's pose ``t`` seconds into the wave, in degrees."""
    pose = [HELLO_DEG[name] for name in JOINTS]
    for name, (amplitude, phase) in WAVE_DEG.items():
        pose[JOINTS.index(name)] += amplitude * math.sin(2.0 * math.pi * (t / PERIOD_S + phase))
    return pose


class Glide:
    """One target per tick from ``start_deg`` to ``goal_deg``, on a straight line in joint space.

    Lasts ``max(min_s, largest joint gap / max_deg_per_s)``, so no joint moves faster than
    ``max_deg_per_s``. The gripper, given as ``(from, to)``, ramps over the same ticks.
    """

    def __init__(
        self,
        start_deg: Sequence[float],
        goal_deg: Sequence[float],
        max_deg_per_s: float,
        rate_hz: float,
        min_s: float = 1.0,
        *,
        gripper: tuple[float, float] | None = None,
    ) -> None:
        self.start = [float(v) for v in start_deg]
        self.goal = [float(v) for v in goal_deg]
        gap = max(abs(g - s) for s, g in zip(self.start, self.goal))
        self.duration_s = max(min_s, gap / max_deg_per_s)
        self.ticks = max(1, math.ceil(self.duration_s * rate_hz - 1e-9))
        self.gripper = gripper
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= self.ticks

    def next(self) -> tuple[list[float], float | None]:
        """The next target: ``(joints_deg, gripper)``, the gripper None when the glide has none."""
        self.i = min(self.i + 1, self.ticks)
        f = self.i / self.ticks
        joints = [s + (g - s) * f for s, g in zip(self.start, self.goal)]
        if self.gripper is None:
            return joints, None
        a, b = self.gripper
        return joints, a + (b - a) * f


class Wave:
    """Rise to the hello pose, wave, then glide back: one ``(joints_deg, gripper)`` target per tick.

    ``phase`` goes ``rise``, ``wave``, ``back``, then ``done``, when ``step()`` returns None.
    """

    def __init__(
        self,
        start_deg: Sequence[float],
        start_gripper: float,
        rate_hz: float,
        *,
        back_deg: Sequence[float],
        back_gripper: float,
    ) -> None:
        self.rate_hz = rate_hz
        self.back_deg = list(back_deg)
        self.back_gripper = back_gripper
        self.phase = "rise"
        self.wave_ticks = round(WAVES * PERIOD_S * rate_hz)
        self._i = 0
        self._last = list(start_deg)
        self._glide = Glide(start_deg, wave_pose(0.0), GLIDE_DEG_PER_S, rate_hz,
                            gripper=(start_gripper, GRIPPER_OPEN))

    def step(self) -> tuple[list[float], float] | None:
        if self.phase == "rise":
            if not self._glide.done:
                return self._keep(self._glide.next())
            self.phase = "wave"
        if self.phase == "wave":
            if self._i < self.wave_ticks:
                t = self._i / self.rate_hz
                self._i += 1
                return self._keep((wave_pose(t), GRIPPER_OPEN))
            self.phase = "back"
            self._glide = Glide(self._last, self.back_deg, GLIDE_DEG_PER_S, self.rate_hz,
                                gripper=(GRIPPER_OPEN, self.back_gripper))
        if self.phase == "back":
            if not self._glide.done:
                return self._keep(self._glide.next())
            self.phase = "done"
        return None

    def _keep(self, target: tuple[list[float], float | None]) -> tuple[list[float], float]:
        joints, gripper = target
        self._last = joints
        return joints, float(gripper if gripper is not None else GRIPPER_OPEN)


def newest_frame(camera: Any, last_seq: int | None) -> tuple[Any, int | None]:
    """``(frame, seq)`` for the camera's newest frame not seen yet, else ``(None, last_seq)``.

    A camera in its own process (the default) shares its frames, numbered; one in the app's process
    is read with ``get_frame()``.
    """
    latest = getattr(camera, "latest_frame", None)
    if callable(latest):
        frame, seq = latest()
        if frame is None or seq == last_seq:
            return None, last_seq
        return frame, seq
    return camera.get_frame(), None


def first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


class WaveHello(App):
    def setup(self) -> None:
        if self.config.robot.type not in SO101:
            raise RuntimeError("Wave hello runs on the SO-101 arm.")
        start = [math.degrees(v) for v in self.robot.get_joint_positions()]
        gripper = min(1.0, max(0.0, float(self.robot.gripper_get_width())))
        self.camera = (getattr(self.robot, "devices", None) or {}).get("camera")
        self.frames_seen = 0
        self._seq: int | None = None
        self._done = False
        self.wave = Wave(start, gripper, self.rate_hz, back_deg=clip_to_envelope(start), back_gripper=gripper)
        log.info("Rising to say hello.")

    @property
    def phase(self) -> str:
        return self.wave.phase

    def loop(self) -> None:
        if self._done:
            return
        before = self.wave.phase
        target = self.wave.step()
        if self.wave.phase != before:
            if self.wave.phase == "wave":
                log.info("Waving.")
            elif self.wave.phase == "back":
                log.info("Done, going back.")
        if target is None:
            self._done = True
            self.finish()
            return
        joints, gripper = target
        self.robot.send_teleop_action([math.radians(v) for v in joints], gripper)
        if self.wave.phase == "wave" and self.camera is not None and self.tick % CAMERA_EVERY_TICKS == 0:
            self._look()

    def on_shutdown(self) -> None:
        if getattr(self, "frames_seen", 0):
            log.info("Saw you in %d camera frames.", self.frames_seen)

    def _look(self) -> None:
        try:
            frame, self._seq = newest_frame(self.camera, self._seq)
        except Exception as exc:  # noqa: BLE001 - the camera is optional: wave without it
            log.warning("The camera stopped answering, so the wave goes on without it (%s).", first_line(exc))
            self.camera = None
            return
        if frame is not None:
            self.frames_seen += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(WaveHello.from_config("robot.yaml").run())
