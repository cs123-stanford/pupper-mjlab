"""Tests for Pupper motion design and the Pupper tracking task's motion files."""

import numpy as np
import pytest

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import DEFAULT_POSE, JOINT_NAMES
from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.env_cfgs import HULL_BODIES


@pytest.fixture(scope="module")
def sit_motion():
  from mjlab.tasks.tracking.config.pupper.tricks import sit

  return md.build_motion(sit.keyframes(), fps=50.0)


def test_ik_reaches_standing_feet_from_a_perturbed_seed():
  k = md.kinematics()
  feet = md.stand_feet()
  root = np.array([0.0, 0.0, md.STAND_HEIGHT])
  quat = np.array([1.0, 0.0, 0.0, 0.0])
  q = np.array(DEFAULT_POSE) + 0.3
  for leg in md.LEGS:
    q, err = k.solve_leg(leg, feet[leg], root, quat, q, iters=200)
    assert err < 1e-4, leg


def test_hulls_add_collision_without_mass():
  k = md.kinematics()
  assert len(k.hull_points) == len(HULL_BODIES)
  assert k.model.body_mass.sum() == pytest.approx(3.17, abs=0.01)


def test_sit_motion_is_clean(sit_motion):
  r = sit_motion.report
  assert r["max_ik_error_m"] < 0.002
  assert r["lowest_foot_bottom_m"] > -0.001
  assert r["lowest_body_point_m"] > -0.001
  assert r["max_joint_speed_rad_s"] < 20.0
  assert r["fraction_of_joint_samples_at_limit"] == 0.0


def test_npz_matches_the_tracking_task_layout(sit_motion, tmp_path):
  path = sit_motion.save_npz(tmp_path / "sit.npz")
  data = np.load(path)
  k = md.kinematics()
  n_frames = len(sit_motion.root_pos)
  n_bodies = k.model.nbody - 1
  assert list(data["body_names"]) == k.body_names
  assert data["joint_pos"].shape == (n_frames, len(JOINT_NAMES))
  assert data["joint_vel"].shape == (n_frames, len(JOINT_NAMES))
  for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
    assert data[key].shape == (n_frames, n_bodies, 3), key
  assert data["body_quat_w"].shape == (n_frames, n_bodies, 4)
  # The anchor body (base_link) follows the designed root exactly.
  base = k.body_names.index("base_link")
  np.testing.assert_allclose(
    data["body_pos_w"][:, base], sit_motion.root_pos, atol=1e-5
  )


def test_csv_round_trip(sit_motion, tmp_path):
  rows = np.loadtxt(sit_motion.save_csv(tmp_path / "sit.csv"), delimiter=",")
  assert rows.shape == (len(sit_motion.root_pos), 3 + 4 + 12)
  np.testing.assert_allclose(rows[:, 7:], sit_motion.joint_pos, atol=1e-5)
  robot = np.genfromtxt(
    sit_motion.save_robot_csv(tmp_path / "r.csv"), delimiter=",", names=True
  )
  np.testing.assert_allclose(
    robot["leg_back_l_3"], sit_motion.joint_pos[:, 11], atol=1e-5
  )


def _still(frames: int, offset: float = 0.0) -> md.Motion:
  """A motion that holds the default stand (plus a joint offset) for some frames."""
  joints = np.tile(np.array(DEFAULT_POSE) + offset, (frames, 1))
  root = np.tile([0.0, 0.0, md.STAND_HEIGHT], (frames, 1))
  quat = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
  return md.Motion(fps=50.0, root_pos=root, root_quat=quat, joint_pos=joints)


def test_concat_clips_shares_boundary_frames_and_points_at_first_use(tmp_path):
  clips = [
    md.Clip("a", _still(10), next="idle"),
    md.Clip("idle", _still(5), loop=True),
    md.Clip("b", _still(8), next="idle"),
  ]
  library = md.concat_clips(clips, order=["a", "idle", "b", "idle"])
  # Each later clip drops its first frame, which equals the previous clip's last.
  assert len(library.root_pos) == 10 + 4 + 7 + 4
  table = {c["name"]: c for c in library.clips}
  assert [c["name"] for c in library.clips] == ["a", "idle", "b"]
  assert (table["a"]["start"], table["a"]["end"]) == (0, 10)
  assert (table["idle"]["start"], table["idle"]["end"]) == (9, 14)
  assert (table["b"]["start"], table["b"]["end"]) == (13, 21)
  assert table["idle"]["loop"] and table["b"]["next"] == "idle"

  path = library.save_npz(tmp_path / "library.npz")
  assert md.read_clips(path) == library.clips


def test_concat_clips_refuses_a_pose_jump():
  with pytest.raises(ValueError, match="rad away"):
    md.concat_clips([md.Clip("a", _still(5)), md.Clip("b", _still(5, offset=0.2))])


def test_single_motion_npz_has_no_clips(sit_motion, tmp_path):
  assert md.read_clips(sit_motion.save_npz(tmp_path / "sit.npz")) == []
