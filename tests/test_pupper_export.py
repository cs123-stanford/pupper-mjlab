"""Parity tests for the Pupper deploy-JSON exporter.

Verifies that the exported RTNeural JSON reproduces the source rsl_rl actor's
output exactly, including the folded observation normalization and the
history frame-reversal (mjlab oldest-first -> robot newest-first).
"""

from typing import Any, cast

import numpy as np
import pytest
import torch
from rsl_rl.modules import MLP, EmpiricalNormalization

from mjlab.tasks.pupper.export import (
  convert_pupper_policy,
  json_forward,
  to_robot_order,
)

HISTORY = 20
SINGLE_OBS = 36
IN_DIM = HISTORY * SINGLE_OBS  # 720
ACTION_DIM = 12
HIDDEN = (256, 128, 128, 128)


class _Actor(torch.nn.Module):
  """Minimal MLPModel-like holder: has ``.mlp`` and ``.obs_normalizer``."""

  def __init__(self, normalize: bool):
    super().__init__()
    self.mlp = MLP(IN_DIM, ACTION_DIM, HIDDEN, activation="elu")
    self.obs_normalizer = (
      EmpiricalNormalization(IN_DIM) if normalize else torch.nn.Identity()
    )

  def forward(self, x):
    return self.mlp(self.obs_normalizer(x))


def _metadata_kwargs() -> dict[str, Any]:
  rng = np.random.default_rng(0)
  return dict(
    action_scale=0.75,
    kp=5.5,
    kd=0.25,
    default_joint_pos=rng.uniform(-0.5, 0.5, ACTION_DIM).tolist(),
    joint_lower_limits=[-2.0] * ACTION_DIM,
    joint_upper_limits=[2.0] * ACTION_DIM,
    observation_history=HISTORY,
    single_obs_dim=SINGLE_OBS,
  )


def _randomize_normalizer(actor: _Actor) -> None:
  # Give the normalizer non-trivial running stats so the fold is exercised.
  normalizer = actor.obs_normalizer
  assert isinstance(normalizer, EmpiricalNormalization)
  with torch.no_grad():
    cast(torch.Tensor, normalizer._mean).copy_(torch.randn(1, IN_DIM))
    cast(torch.Tensor, normalizer._std).copy_(torch.rand(1, IN_DIM) + 0.5)


def test_export_parity_with_normalization():
  torch.manual_seed(1)
  actor = _Actor(normalize=True).eval()
  _randomize_normalizer(actor)

  policy = convert_pupper_policy(actor, **_metadata_kwargs())

  x = torch.randn(8, IN_DIM)  # mjlab (oldest-first) order.
  with torch.no_grad():
    expected = actor(x).cpu().numpy()

  got = json_forward(policy, to_robot_order(x, HISTORY, SINGLE_OBS))
  np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_export_parity_without_normalization():
  torch.manual_seed(2)
  actor = _Actor(normalize=False).eval()

  policy = convert_pupper_policy(actor, **_metadata_kwargs())

  x = torch.randn(4, IN_DIM)
  with torch.no_grad():
    expected = actor(x).cpu().numpy()
  got = json_forward(policy, to_robot_order(x, HISTORY, SINGLE_OBS))
  np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_frame_reversal_actually_reorders():
  # Without the reversal, feeding robot-order obs to a policy folded for
  # mjlab-order must NOT match -- proves the reversal is load-bearing.
  torch.manual_seed(3)
  actor = _Actor(normalize=True).eval()
  _randomize_normalizer(actor)
  policy = convert_pupper_policy(actor, **_metadata_kwargs())

  x = torch.randn(4, IN_DIM)
  with torch.no_grad():
    expected = actor(x).cpu().numpy()
  # Feed mjlab order directly (no reversal): should differ.
  wrong = json_forward(policy, x.cpu().numpy())
  assert not np.allclose(wrong, expected, rtol=1e-3, atol=1e-3)


def test_schema_shape():
  actor = _Actor(normalize=True).eval()
  policy = convert_pupper_policy(actor, **_metadata_kwargs())

  assert policy["in_shape"] == [None, IN_DIM]
  assert policy["observation_history"] == HISTORY
  assert policy["use_imu"] is True
  assert policy["control_orientation"] is True
  # 5 dense layers: elu x4 then linear (empty activation).
  assert len(policy["layers"]) == len(HIDDEN) + 1
  assert [ly["activation"] for ly in policy["layers"]] == ["elu"] * 4 + [""]
  # Final layer outputs the 12 action dims.
  assert policy["layers"][-1]["shape"] == [None, ACTION_DIM]
  # Kernel is [in][out]: first layer is 720 rows x 256 cols.
  w0 = policy["layers"][0]["weights"][0]
  assert len(w0) == IN_DIM and len(w0[0]) == HIDDEN[0]


def test_contract_v2_obs_spec_and_command_clip():
  actor = _Actor(normalize=True).eval()
  policy = convert_pupper_policy(
    actor,
    **_metadata_kwargs(),
    command_clip={"vx": (-0.49, 0.49), "vy": (-0.3, 0.3), "wz": (-2.0, 2.0)},
  )

  assert policy["contract_version"] == 2
  spec = policy["obs_spec"]
  assert spec["frame_order"] == "newest_first"
  assert spec["history"] == HISTORY
  assert sum(c["size"] for c in spec["components"]) == SINGLE_OBS
  assert [c["name"] for c in spec["components"]] == [
    "base_ang_vel",
    "projected_gravity",
    "command",
    "desired_world_z",
    "joint_pos_rel",
    "last_action",
  ]
  assert policy["command_clip"] == {
    "vx": [-0.49, 0.49],
    "vy": [-0.3, 0.3],
    "wz": [-2.0, 2.0],
  }


def test_obs_spec_must_sum_to_frame_size():
  actor = _Actor(normalize=True).eval()
  with pytest.raises(ValueError, match="components sum"):
    convert_pupper_policy(
      actor,
      **_metadata_kwargs(),
      obs_components=(("base_ang_vel", 3), ("command", 3)),
    )


def test_command_clip_rejects_inverted_bounds():
  actor = _Actor(normalize=True).eval()
  with pytest.raises(ValueError, match="inverted"):
    convert_pupper_policy(actor, **_metadata_kwargs(), command_clip={"vx": (0.5, -0.5)})
