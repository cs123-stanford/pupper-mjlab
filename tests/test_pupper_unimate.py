"""The UniMate -> Pupper fit, without UniMate: a still skeleton fits a still stand."""

from __future__ import annotations

import numpy as np

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper import unimate as U


def _standing(frames: int) -> np.ndarray:
  """Pupper's rig at rest, as UniMate returns it (lowest joint at z = 0)."""
  rest = np.array(U.SKELETON["rest"])
  rest[:, 2] -= rest[:, 2].min()
  return np.repeat(rest[None], frames, axis=0)


def test_skeleton_is_a_tree_with_four_legs():
  sk = U.SKELETON
  assert len(sk["names"]) == len(sk["clean"]) == len(sk["parents"]) == 19
  assert sk["parents"][0] == -1
  assert all(0 <= p < j for j, p in enumerate(sk["parents"]) if j > 0)
  for root, foot in sk["legs"].values():
    assert root in sk["names"] and foot in sk["names"]


def test_still_skeleton_fits_the_default_stand():
  fit = U.fit_pupper(_standing(U.SEGMENT))
  m = fit.motion
  assert m.fps == U.FPS
  assert len(m.root_pos) == 99  # 2 s at 50 Hz (60 frames at 30 fps)
  assert np.abs(m.root_pos[:, 2] - md.STAND_HEIGHT).max() < 0.01
  assert np.abs(m.joint_pos - np.array(md.DEFAULT_POSE)).max() < 0.1
  assert fit.stats["ik_foot_error_worst_cm"] < 0.5
  assert fit.stats["frames_with_no_foot_down"] == 0


def test_timing_options():
  raw = _standing(U.SEGMENT)
  assert len(U.fit_pupper(raw, slow=2.0).motion.root_pos) == 197
  assert len(U.fit_pupper(raw, hold=1.0).motion.root_pos) == 149
  assert len(U.fit_pupper(raw, trim_start=0.5, trim_end=0.5).motion.root_pos) == 49
  # Three chained prompts (60 + 50 + 50 frames), each stretched to 3 s.
  chained = _standing(U.SEGMENT + 2 * (U.SEGMENT - 10))
  fit = U.fit_pupper(chained, durations=[3.0, 3.0, 3.0])
  assert abs(fit.motion.duration - 7.96) < 0.05
