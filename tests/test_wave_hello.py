"""Wave hello: the motion rules on a fake arm, the simulated follower, the camera, and the built package."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import pytest
from conftest import (
    CUT_GOAL,
    MID_DEG,
    REST_DEG,
    SIM_SPEEDUP,
    SINGLE_GOAL,
    FakeArm,
    FakeCamera,
    WithDevices,
    assert_motion_ok,
    built_manifest,
    envelope_deg,
    make_app,
    run_paced,
    run_ticks,
    run_under_sdk,
    sim_joints_deg,
    urdf_limits_deg,
    wait_until_still,
)

VERSION = "0.2.0"
WHATS_NEW = "Waves both ways."
RATE_HZ = 50
MAX_TICKS = 5000
STARTS = {"rest": REST_DEG, "mid": MID_DEG}


def run_whole(main, arm: FakeArm, robot_type: str = "hf_so101_follower"):
    app = make_app(main, arm, robot_type)
    run_ticks(app, MAX_TICKS)
    return app


def phase_runs(arm: FakeArm) -> list[tuple[str, int]]:
    """The phases the targets were sent in, in order, with how many targets each."""
    runs: list[tuple[str, int]] = []
    for label in arm.labels():
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return runs


# --------------------------------------------------------------------------- motion


@pytest.mark.parametrize("start", sorted(STARTS))
def test_motion_follows_the_rules(main, start):
    arm = FakeArm(STARTS[start], gripper=0.1)
    app = run_whole(main, arm)
    assert app.rate_hz == RATE_HZ
    assert_motion_ok(arm, RATE_HZ)
    assert {method for method, *_ in arm.targets} == {"send_teleop_action"}


@pytest.mark.parametrize("start", sorted(STARTS))
def test_phases_wave_length_and_one_finish(main, start):
    arm = FakeArm(STARTS[start])
    app = run_whole(main, arm)
    runs = phase_runs(arm)
    assert [name for name, _ in runs] == ["rise", "wave", "back"]
    wave_ticks = dict(runs)["wave"]
    assert abs(wave_ticks - main.WAVES * main.PERIOD_S * RATE_HZ) <= 2
    assert len(app.finish_calls) == 1
    assert app.finish_calls[0] == len(arm.targets), "finish() comes in the tick after the back glide"
    assert app.phase == "done"


@pytest.mark.parametrize("start", sorted(STARTS))
def test_goes_back_to_the_start_clipped_into_the_envelope(main, start):
    arm = FakeArm(STARTS[start], gripper=0.25)
    run_whole(main, arm)
    _, joints, gripper, _ = arm.targets[-1]
    assert joints == pytest.approx(main.clip_to_envelope(STARTS[start]), abs=1e-9)
    assert all(lo <= v <= hi for v, (lo, hi) in zip(joints, envelope_deg()))
    assert gripper == pytest.approx(0.25)
    if start == "mid":
        assert joints == pytest.approx(list(MID_DEG), abs=1e-9)


def test_the_app_envelope_sits_inside_the_urdf_one(main):
    for name, (lo, hi), (ulo, uhi) in zip(main.JOINTS, (main.ENVELOPE_DEG[j] for j in main.JOINTS), envelope_deg()):
        assert ulo - 1e-9 <= lo <= ulo + 1.0, name
        assert uhi - 1.0 <= hi <= uhi + 1e-9, name


def test_the_motion_check_catches_a_target_past_the_urdf_limits():
    """The SDK's safety check has no SO-101 limits, so ``assert_motion_ok`` checks the URDF ones
    itself: a small step inward that is still past the shoulder's limit fails."""
    lowest = urdf_limits_deg()[1][0]
    arm = FakeArm([0.0, lowest - 6.0, 30.0, 0.0, 0.0])
    arm.send_teleop_action([math.radians(v) for v in (0.0, lowest - 5.5, 30.0, 0.0, 0.0)], 0.0)
    with pytest.raises(AssertionError, match="URDF"):
        assert_motion_ok(arm, RATE_HZ)


def test_the_tests_never_see_the_real_home(tmp_path):
    """CONTRACTS 0.5: HOME and MONOROBOT_HOME both point under the test's temp folder."""
    assert Path.home().resolve().is_relative_to(tmp_path.resolve())
    assert Path(os.environ["MONOROBOT_HOME"]).resolve().is_relative_to(tmp_path.resolve())


def test_whole_run_is_about_ten_seconds_from_rest(main):
    arm = FakeArm(REST_DEG)
    run_whole(main, arm)
    seconds = len(arm.targets) / RATE_HZ
    assert 8.0 <= seconds <= 12.0, seconds


def test_waves_both_ways(main):
    """The wrist nods and turns while the base swings side to side, a quarter period apart."""
    arm = FakeArm(MID_DEG)
    run_whole(main, arm)
    wave = [joints for _, joints, _, label in arm.targets if label == "wave"]
    hello = [main.HELLO_DEG[name] for name in main.JOINTS]
    moving = {"shoulder_pan", "wrist_flex", "wrist_roll"}
    for i, name in enumerate(main.JOINTS):
        values = [pose[i] for pose in wave]
        if name in moving:
            assert max(values) == pytest.approx(hello[i] + 15.0, abs=0.1), name
            assert min(values) == pytest.approx(hello[i] - 15.0, abs=0.1), name
        else:
            assert values == pytest.approx([hello[i]] * len(values)), name
    pan, flex = main.JOINTS.index("shoulder_pan"), main.JOINTS.index("wrist_flex")
    assert wave[0][pan] == pytest.approx(hello[pan] + 15.0)      # the base starts at one side
    assert wave[0][flex] == pytest.approx(hello[flex])           # while the wrist is level
    quarter = round(main.PERIOD_S * RATE_HZ / 4)
    assert wave[quarter][pan] == pytest.approx(hello[pan], abs=0.5)
    assert wave[quarter][flex] == pytest.approx(hello[flex] + 15.0, abs=0.5)


def test_nothing_moves_in_setup(main):
    arm = FakeArm(REST_DEG)
    app = make_app(main, arm)
    app.setup()
    assert arm.targets == []
    assert arm.reads["get_joint_positions"] == 1


def test_a_gripper_reading_just_past_open_is_clamped(main):
    arm = FakeArm(MID_DEG, gripper=1.0)
    arm.gripper = 1.02
    arm.start_gripper = 1.0
    run_whole(main, arm)
    assert_motion_ok(arm, RATE_HZ)


def test_every_tick_sends_one_target(main):
    """One target per tick and no blocking move: a Stop lands between any two ticks."""
    arm = FakeArm(REST_DEG)
    app = make_app(main, arm)
    run_ticks(app, 150)
    assert len(arm.targets) == 150
    assert app.finish_calls == []


def test_needs_an_so101(main):
    app = make_app(main, FakeArm(), robot_type="lynxmotion_al5d")
    with pytest.raises(RuntimeError) as exc:
        app.setup()
    assert str(exc.value) == "Wave hello runs on the SO-101 arm."


def test_logs_are_short_and_plain(main, caplog):
    caplog.set_level(logging.INFO, logger="wave-hello")
    run_whole(main, FakeArm(REST_DEG))
    lines = [r.getMessage() for r in caplog.records if r.name == "wave-hello"]
    assert lines == ["Rising to say hello.", "Waving.", "Done, going back."]
    for line in lines:
        assert line[0].isupper() and "—" not in line and "so101" not in line.lower()


# --------------------------------------------------------------------------- camera


def test_runs_without_a_camera(main, caplog):
    caplog.set_level(logging.INFO, logger="wave-hello")
    arm = FakeArm(REST_DEG)
    app = run_whole(main, arm)
    app.on_shutdown()
    assert app.frames_seen == 0
    assert not any("camera" in r.getMessage() for r in caplog.records)


def test_counts_camera_frames_while_waving(main, caplog):
    caplog.set_level(logging.INFO, logger="wave-hello")
    camera = FakeCamera()
    arm = FakeArm(REST_DEG, devices={"camera": camera})
    app = run_whole(main, arm)
    app.on_shutdown()
    expected = math.ceil(main.WAVES * main.PERIOD_S * RATE_HZ / main.CAMERA_EVERY_TICKS)
    assert abs(app.frames_seen - expected) <= 1
    assert camera.reads == app.frames_seen
    assert f"Saw you in {app.frames_seen} camera frames." in [r.getMessage() for r in caplog.records]
    assert_motion_ok(arm, RATE_HZ)


def test_a_camera_that_stops_answering_is_left_out(main, caplog):
    caplog.set_level(logging.INFO, logger="wave-hello")
    arm = FakeArm(REST_DEG, devices={"camera": FakeCamera(fail_after=3)})
    app = run_whole(main, arm)
    assert app.frames_seen == 3
    assert len(app.finish_calls) == 1
    assert any("camera stopped answering" in r.getMessage() for r in caplog.records)


def test_an_in_process_camera_is_read_with_get_frame(main):
    class InProcessCamera:
        def __init__(self):
            self.calls = 0

        def get_frame(self):
            self.calls += 1
            return [[0]]

    camera = InProcessCamera()
    app = run_whole(main, FakeArm(MID_DEG, devices={"camera": camera}))
    assert camera.calls == app.frames_seen > 0


# --------------------------------------------------------------------------- kinematics


@pytest.mark.parametrize("start", sorted(STARTS))
def test_the_gripper_stays_10_cm_above_the_base(main, start):
    """Forward kinematics on the vendored URDF (ikpy): the gripper frame never comes within 10 cm of the base plane."""
    pytest.importorskip("ikpy")
    from monorobot.hal.adapters.hf_so101.follower import _URDF_PATH, _URDF_TARGET_FRAME, ARM_MOTOR_NAMES
    from monorobot.hal.kinematics import URDFChainKinematics

    kin = URDFChainKinematics(_URDF_PATH, ARM_MOTOR_NAMES, target_frame=_URDF_TARGET_FRAME)
    arm = FakeArm(STARTS[start])
    run_whole(main, arm)
    poses = [arm.start_deg] + [joints for _, joints, _, _ in arm.targets]
    heights = [kin.fk([math.radians(v) for v in pose])[2, 3] for pose in poses]
    assert min(heights) >= 0.10, min(heights)


# --------------------------------------------------------------------------- simulated follower


def test_on_the_simulated_follower(main, sim_follower):
    start = sim_joints_deg(sim_follower)
    assert start == pytest.approx(list(REST_DEG), abs=0.01)     # a new simulated arm starts folded
    app = make_app(main, sim_follower, robot_type="sim_so101_follower")
    run_paced(app, MAX_TICKS)
    assert len(app.finish_calls) == 1
    clipped = main.clip_to_envelope(start)
    joints = wait_until_still(sim_follower, clipped)
    assert joints == pytest.approx(clipped, abs=2.0)             # before the rest fold
    sim_follower.disconnect()                                    # on_exit: rest folds the last part
    state = json.loads(Path(sim_follower.state_file).read_text())
    assert state["joints_deg"] == pytest.approx(list(REST_DEG), abs=0.5)
    assert state["torque"] == "off"


def test_the_simulated_follower_reports_a_jump(tmp_path, caplog):
    """The guard the simulated runs above rely on: a goal 30 degrees from the arm is reported.

    The follower keeps every goal within 10 degrees of the arm, so it never sends the jump (no
    "single goal"): it cuts it, says so ("was cut to") and counts it in ``cut_goals``. An engine from
    before that cut sent it whole, logged "single goal" and counted it in ``big_goals``.
    """
    pytest.importorskip("monorobot.hal.adapters.sim_so101")
    from monorobot.hal.registry import load_adapter

    arm = load_adapter("sim_so101_follower", isolate=False, robot_id="sim-jump", state_dir=tmp_path / "sim",
                       time_scale=SIM_SPEEDUP)
    assert arm.connect()
    try:
        with caplog.at_level(logging.WARNING):
            arm.send_teleop_action([math.radians(v) for v in (30.0, *REST_DEG[1:])], 0.0)
        messages = [r.getMessage() for r in caplog.records]
        if hasattr(arm, "cut_goals"):
            assert (arm.cut_goals, arm.big_goals) == (1, 0)
            assert any(CUT_GOAL in m for m in messages) and not any(SINGLE_GOAL in m for m in messages)
        else:
            assert arm.big_goals == 1
            assert any(SINGLE_GOAL in m for m in messages)
    finally:
        arm.emergency_stop()
        arm.disconnect()


def test_on_the_simulated_follower_with_a_camera(main, sim_follower):
    camera = FakeCamera()
    app = make_app(main, WithDevices(sim_follower, {"camera": camera}), robot_type="sim_so101_follower")
    run_paced(app, MAX_TICKS)
    assert app.frames_seen > 0


def test_the_sdk_runs_it_to_the_end(main, tmp_path, monkeypatch, caplog):
    """The SDK's own run(), at the app's own rate: finish() ends the run with exit code 0, then the
    arm's on_exit folds it to rest and lets go. One wave, to keep it short."""
    caplog.set_level(logging.INFO, logger="wave-hello")
    monkeypatch.setattr(main, "WAVES", 1)
    camera = FakeCamera()
    app, code, state = run_under_sdk(main, tmp_path, devices={"camera": camera})
    assert code == 0 and app.error is None
    assert app.phase == "done"
    assert app.frames_seen == camera.reads > 0
    assert not camera.connected
    assert state["joints_deg"] == pytest.approx(list(REST_DEG), abs=0.5)
    assert state["torque"] == "off"
    lines = [r.getMessage() for r in caplog.records if r.name == "wave-hello"]
    assert lines == ["Rising to say hello.", "Waving.", "Done, going back.",
                     f"Saw you in {app.frames_seen} camera frames."]


# --------------------------------------------------------------------------- the package


def test_built_manifest(tmp_path):
    manifest, names = built_manifest(tmp_path)
    assert manifest["schema"] == "monorobot/1.0"
    assert manifest["name"] == "wave-hello"
    assert manifest["version"] == VERSION
    assert manifest["author"] == {"handle": "grand-wizard-create"}
    assert manifest["compat"]["robots"] == ["hf_so101_follower"]
    assert manifest["devices"] == {"required": [], "optional": ["camera"]}
    assert manifest["permissions"]["network"] is False
    assert manifest["permissions"]["filesystem_write"] == []
    assert manifest["license"] == "Apache-2.0"
    assert manifest["whats_new"] == WHATS_NEW
    assert manifest["description"] == "Your arm waves hello, then folds back to rest. A gentle first app."
    assert "also_uses" not in manifest
    assert sorted(names) == [".gitignore", "README.md", "main.py", "manifest.json", "pyproject.toml", "robot.yaml"]
