"""Parity tests for the Pupper gait deploy export.

Two things have to hold for the gait policy to survive the trip to the robot:

1. The 48-dim frame (36 proprio + 12 reference offset) folds and frame-reverses
   just like the 36-dim velocity frame did.
2. The reference offset the robot computes itself matches the one the policy was
   trained against -- in particular the time-reversed (backward / turn-left) case,
   where the phase goes negative and only a floor-modulo wraps it correctly.
"""

from typing import cast

import numpy as np
import pytest
import torch
from rsl_rl.modules import MLP, EmpiricalNormalization

from mjlab.tasks.pupper.export import (
  convert_pupper_policy,
  json_forward,
  to_robot_order,
)
from mjlab.tasks.pupper_gait.export import (
  GAIT_OBS_COMPONENTS,
  GAIT_REFERENCE_IDX,
  GAIT_SINGLE_OBS_DIM,
  gait_reference_offset_torch,
  reference_offset_numpy,
)
from mjlab.tasks.pupper_gait.mdp.gait_reference import build_joint_reference_table

HISTORY = 20
IN_DIM = HISTORY * GAIT_SINGLE_OBS_DIM  # 960
ACTION_DIM = 12
HIDDEN = (256, 128, 128, 128)

N_SAMPLES = 25  # Coarser than the deployed 100 to keep the IK cheap in tests.


class _Actor(torch.nn.Module):
  """Minimal MLPModel-like holder: has ``.mlp`` and ``.obs_normalizer``."""

  def __init__(self):
    super().__init__()
    self.mlp = MLP(IN_DIM, ACTION_DIM, HIDDEN, activation="elu")
    self.obs_normalizer = EmpiricalNormalization(IN_DIM)

  def forward(self, x):
    return self.mlp(self.obs_normalizer(x))


def _actor() -> _Actor:
  torch.manual_seed(1)
  actor = _Actor().eval()
  with torch.no_grad():
    cast(torch.Tensor, actor.obs_normalizer._mean).copy_(torch.randn(1, IN_DIM))
    cast(torch.Tensor, actor.obs_normalizer._std).copy_(torch.rand(1, IN_DIM) + 0.5)
  return actor


def _policy(actor: _Actor) -> dict:
  rng = np.random.default_rng(0)
  return convert_pupper_policy(
    actor,
    action_scale=0.75,
    kp=5.5,
    kd=0.25,
    default_joint_pos=rng.uniform(-0.5, 0.5, ACTION_DIM).tolist(),
    joint_lower_limits=[-2.0] * ACTION_DIM,
    joint_upper_limits=[2.0] * ACTION_DIM,
    observation_history=HISTORY,
    single_obs_dim=GAIT_SINGLE_OBS_DIM,
    obs_components=GAIT_OBS_COMPONENTS,
  )


def _gait_block() -> dict:
  return {
    "n_samples": N_SAMPLES,
    "frequency": 1.0 / 0.75,
    "blend_speed": 0.1,
    "gallop_speed": 0.5,
    "gallop_freq_mult": 2.0,
    "dir_threshold": 0.05,
    "reference_index": GAIT_REFERENCE_IDX,
    "joint_names": [f"j{i}" for i in range(ACTION_DIM)],
    "trot_table": build_joint_reference_table(N_SAMPLES, "trot").tolist(),
    "gallop_table": build_joint_reference_table(N_SAMPLES, "gallop").tolist(),
  }


def _commands() -> np.ndarray:
  """(t, vx, vy, yaw) cases spanning stand / trot / gallop / reversal."""
  rng = np.random.default_rng(7)
  cases = [
    (0.0, 0.0, 0.0, 0.0),  # standing: zero blend, zero offset.
    (0.31, 0.3, 0.0, 0.0),  # forward trot.
    (0.31, -0.3, 0.0, 0.0),  # backward: time-reversed phase.
    (0.87, 0.02, 0.0, 0.8),  # turn left, |vx| under the direction threshold.
    (0.87, 0.02, 0.0, -0.8),  # turn right: reversed via yaw sign.
    (1.23, 0.6, 0.0, 0.0),  # gallop (|vx| >= gallop_speed, 2x clock).
    (1.23, -0.9, 0.1, 0.2),  # backward gallop.
    (0.5, 0.03, 0.01, 0.0),  # tiny command: blend still saturated by norm/0.1.
    (0.5, 0.001, 0.0, 0.0),  # sub-blend_speed command: partial blend.
    (600.0, 0.4, 0.0, 0.0),  # long uptime (no episode resets on the robot).
  ]
  cases += [
    tuple(x)
    for x in rng.uniform([-0.0, -1.0, -0.5, -2.0], [50.0, 1.0, 0.5, 2.0], size=(32, 4))
  ]
  return np.asarray(cases, dtype=np.float64)


def test_export_parity_with_48_dim_frame():
  actor = _actor()
  policy = _policy(actor)

  x = torch.randn(8, IN_DIM)  # mjlab (oldest-first) order.
  with torch.no_grad():
    expected = actor(x).cpu().numpy()

  got = json_forward(policy, to_robot_order(x, HISTORY, GAIT_SINGLE_OBS_DIM))
  np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_schema_advertises_frame_size():
  policy = _policy(_actor())
  assert policy["in_shape"] == [None, IN_DIM]
  assert policy["single_observation_size"] == GAIT_SINGLE_OBS_DIM == 48
  assert policy["observation_history"] == HISTORY
  assert GAIT_REFERENCE_IDX == 36


def test_deploy_reference_offset_matches_training():
  gait = _gait_block()
  default = np.asarray(build_joint_reference_table(N_SAMPLES, "trot")[0])
  cases = _commands()
  t, vx, vy, yaw = cases[:, 0], cases[:, 1], cases[:, 2], cases[:, 3]

  deploy = reference_offset_numpy(gait, default, t, vx, vy, yaw)
  train = gait_reference_offset_torch(
    gait,
    torch.tensor(default, dtype=torch.float32),
    torch.tensor(t, dtype=torch.float32),
    torch.tensor(vx, dtype=torch.float32),
    torch.tensor(vy, dtype=torch.float32),
    torch.tensor(yaw, dtype=torch.float32),
  ).numpy()

  np.testing.assert_allclose(deploy, train, rtol=1e-4, atol=1e-4)


def test_zero_command_gives_zero_offset():
  gait = _gait_block()
  default = np.zeros(ACTION_DIM)
  offset = reference_offset_numpy(
    gait, default, np.array([1.7]), np.zeros(1), np.zeros(1), np.zeros(1)
  )
  np.testing.assert_allclose(offset, 0.0)


def test_time_reversal_is_load_bearing():
  # A backward command must not produce the same reference as a forward one at
  # the same |speed| and phase -- if it does, the reversal was dropped.
  gait = _gait_block()
  default = np.zeros(ACTION_DIM)
  t = np.array([0.31, 0.31])
  fwd = reference_offset_numpy(
    gait, default, t, np.array([0.3, -0.3]), np.zeros(2), np.zeros(2)
  )
  assert not np.allclose(fwd[0], fwd[1], atol=1e-3)


def test_gallop_switch_changes_reference():
  gait = _gait_block()
  default = np.zeros(ACTION_DIM)
  speeds = np.array([gait["gallop_speed"] - 1e-3, gait["gallop_speed"]])
  out = reference_offset_numpy(
    gait, default, np.array([0.4, 0.4]), speeds, np.zeros(2), np.zeros(2)
  )
  assert not np.allclose(out[0], out[1], atol=1e-3)


def _mixed_gait_block() -> dict:
  """A mixed-reference block: the single-gait one plus the lift table.

  Skips (rather than fails) the tests that need it until the lift gait's
  TODO(student) block in ``mdp/gait_reference.py`` is filled in -- once it is,
  these tests un-skip and check that the gait deploys correctly.
  """
  gait = _gait_block()
  try:
    lift = build_joint_reference_table(N_SAMPLES, "lift")
  except NotImplementedError as e:
    pytest.skip(str(e))
  gait["lift_table"] = lift.tolist()
  return gait


def test_lift_table_plays_when_not_translating():
  # A pure turn must play the lift table when it is present; the legacy path
  # (no lift table) trots through the same command.
  default = np.zeros(ACTION_DIM)
  t = np.array([0.4])
  vx, vy, yaw = np.array([0.0]), np.array([0.0]), np.array([0.9])
  lifted = reference_offset_numpy(_mixed_gait_block(), default, t, vx, vy, yaw)
  trotted = reference_offset_numpy(_gait_block(), default, t, vx, vy, yaw)
  assert not np.allclose(lifted, trotted, atol=1e-3)


def test_lift_table_does_not_touch_translating_commands():
  # Forward/backward commands select the trot (or fast) table whether or not a
  # lift table is present -- its presence must not perturb them.
  default = np.zeros(ACTION_DIM)
  t = np.array([0.31, 0.31, 1.23])
  vx = np.array([0.3, -0.3, 0.6])  # trot fwd, trot bwd, fast branch.
  vy = np.zeros(3)
  yaw = np.zeros(3)
  with_lift = reference_offset_numpy(_mixed_gait_block(), default, t, vx, vy, yaw)
  without = reference_offset_numpy(_gait_block(), default, t, vx, vy, yaw)
  np.testing.assert_allclose(with_lift, without, rtol=1e-6, atol=1e-6)


def _direction_split_block() -> dict:
  """A mixed block with a distinct backward-fast capture (direction split)."""
  gait = _mixed_gait_block()
  # Any table that differs from the forward one works; a shifted trot does.
  # (Synthetic on purpose: the generator only ships trot and lift, so every
  # other gait name falls back to the trot and would not differ.)
  back = build_joint_reference_table(N_SAMPLES, "trot") + 0.1
  gait["gallop_back_table"] = back.tolist()
  return gait


def test_fast_back_table_plays_only_backward_fast():
  # Backward-fast must switch to the backward capture; every other branch
  # (forward fast, trot both ways, turning) must be untouched by its presence.
  default = np.zeros(ACTION_DIM)
  t = np.full(5, 1.23)
  vx = np.array([-0.9, 0.9, 0.3, -0.3, 0.0])
  vy = np.zeros(5)
  yaw = np.array([0.0, 0.0, 0.0, 0.0, 0.9])
  split = reference_offset_numpy(_direction_split_block(), default, t, vx, vy, yaw)
  merged = reference_offset_numpy(_mixed_gait_block(), default, t, vx, vy, yaw)
  assert not np.allclose(split[0], merged[0], atol=1e-3)
  np.testing.assert_allclose(split[1:], merged[1:], rtol=1e-6, atol=1e-6)


def test_fast_back_equal_to_forward_is_inert():
  # Shipping the forward table in the backward slot must reproduce the merged
  # (single fast table) behavior exactly -- the split is purely a table swap.
  gait = _mixed_gait_block()
  gait["gallop_back_table"] = gait["gallop_table"]
  default = np.zeros(ACTION_DIM)
  cases = _commands()
  t, vx, vy, yaw = cases[:, 0], cases[:, 1], cases[:, 2], cases[:, 3]
  with_split = reference_offset_numpy(gait, default, t, vx, vy, yaw)
  without = reference_offset_numpy(_mixed_gait_block(), default, t, vx, vy, yaw)
  np.testing.assert_allclose(with_split, without, rtol=1e-6, atol=1e-6)
