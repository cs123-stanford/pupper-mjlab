"""Single-jump reference: crouch, one pronk cycle, hold the landing pose.

Unlike the periodic gait references, the jump plays exactly once per episode:
the reference holds the jump table's phase-0 pose (the crouch) for
``crouch_hold_s`` so the freshly-reset robot can settle into it, sweeps through
one cycle of the ``jump`` gait (the drive-shaped pronk, see
:mod:`gait_reference`), and then holds the touchdown pose -- which is the same
crouch -- for the landing. There is no command blending: the commanded twist is
zero (the jump is in place), so the reference is always at full strength.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.pupper.mdp.observations import PupperObsNoise, pupper_actor_obs
from mjlab.tasks.pupper_gait.mdp import gait_reference as gait_reference_module
from mjlab.tasks.pupper_gait.mdp.gait import _lookup, _reference_table
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_DEFAULT_NOISE = PupperObsNoise()


# Where in the jump cycle the one-shot playback starts and ends.
#
# For the IK pronk, phase 0 is the *touchdown* keyframe -- all four feet
# planted 0.13 m forward of the hips -- fine as a dynamic landing instant but
# statically unstable as a held pose: the robot sits back off its forward
# feet and tips over before launch. Mid-stance (keyframe 2 of 10) has the
# feet directly under the hips at the crouch depth, so the hold, the launch
# and the landing all pass through it.
#
# The captured jump (see gait_reference._CAPTURED_JUMP_FILE) is built with
# row 0 as its own stable hold pose, so playback starts at 0 there.
_JUMP_PHASE_START = 0.0 if gait_reference_module._CAPTURED_JUMP_FILE.exists() else 0.2


def jump_reference_offset(
  env: "ManagerBasedRlEnv",
  frequency: float,
  n_samples: int,
  crouch_hold_s: float,
  phase_lead_steps: float = 0.0,
) -> torch.Tensor:
  """(N, 12) reference joint offset from default for the single-jump schedule.

  The phase is clamped rather than periodic: the mid-stance crouch (see
  ``_JUMP_PHASE_START``) until ``crouch_hold_s``, one full cycle at
  ``frequency``, then pinned one wrap later -- the same crouch -- for the
  landing. The lookup's floor-modulo handles phases past 1.
  """
  robot: Entity = env.scene["robot"]
  default = robot.data.default_joint_pos
  assert default is not None
  t = (env.episode_length_buf.float() - phase_lead_steps) * env.step_dt
  phase = _JUMP_PHASE_START + torch.clamp(
    (t - crouch_hold_s) * frequency, min=0.0, max=1.0
  )
  ref = _lookup(_reference_table(env, n_samples, "jump"), phase, n_samples)
  return ref - default


def jump_reference_obs(
  env: "ManagerBasedRlEnv",
  frequency: float,
  n_samples: int,
  crouch_hold_s: float,
) -> torch.Tensor:
  """Reference pose motion the policy tracks (12 = ref_offset from default)."""
  return jump_reference_offset(env, frequency, n_samples, crouch_hold_s)


def jump_actor_obs(
  env: "ManagerBasedRlEnv",
  command_name: str = "twist",
  add_noise: bool = True,
  frequency: float = 2.0,
  n_samples: int = 100,
  crouch_hold_s: float = 0.3,
  noise: PupperObsNoise = _DEFAULT_NOISE,
  asset_cfg: SceneEntityCfg = _ROBOT,
  imu_latency_distribution: Sequence[float] | None = None,
) -> torch.Tensor:
  """36-dim proprio frame (incl. the zero twist command) + 12 reference = 48.

  Same layout as the gait tasks' actor frame, so the deploy stack's reference
  plumbing carries over once the controller can play a one-shot table.
  """
  base = pupper_actor_obs(
    env, command_name, add_noise, noise, asset_cfg, imu_latency_distribution
  )
  ref = jump_reference_offset(env, frequency, n_samples, crouch_hold_s)
  return torch.cat([base, ref], dim=-1)


class jump_tracking:
  """Reward for matching the jump reference, exp(-mean(error^2/std^2)).

  The tolerance is phase-dependent when ``std_hold`` is given: strict outside
  the playback window (the hold before launch and the settled recovery after
  landing, where the reference is the static crouch and the pose IS the task)
  and loose inside it (the jump itself, where the policy is free to out-jump
  the template). One std for the whole episode lets the policy pre-position
  during the hold -- run e9194f0a parked in a splayed crab crouch under the
  loose gallop std and converted that geometry into launch energy the moment
  the reward gate opened. The reward gate cannot price a *pose*; this does.

  The loose window is ``[loose_from_s, loose_until_s)`` of episode time:
  launch through landing plus a settle grace, thread the same clock values as
  the reference schedule.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.ids = cfg.params["asset_cfg"].joint_ids
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    def _resolve(spec) -> torch.Tensor:
      _, _, vals = resolve_matching_names_values(data=spec, list_of_strings=joint_names)
      return torch.tensor(vals, device=env.device, dtype=torch.float32)

    self.std = _resolve(cfg.params["std"])
    std_hold = cfg.params.get("std_hold")
    self.std_hold = None if std_hold is None else _resolve(std_hold)
    self.loose_from_s = float(cfg.params.get("loose_from_s", 0.0))
    self.loose_until_s = float(cfg.params.get("loose_until_s", float("inf")))
    self.frequency = float(cfg.params["frequency"])
    self.n = int(cfg.params["n_samples"])
    self.crouch_hold_s = float(cfg.params["crouch_hold_s"])
    self.phase_lead_steps = float(cfg.params.get("phase_lead_steps", 0.0))
    _reference_table(env, self.n, "jump")  # warm the cache at construction

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    std,
    asset_cfg: SceneEntityCfg,
    frequency: float,
    n_samples: int,
    crouch_hold_s: float,
    phase_lead_steps: float = 0.0,
    std_hold=None,
    loose_from_s: float = 0.0,
    loose_until_s: float = float("inf"),
  ) -> torch.Tensor:
    del std, frequency, n_samples, crouch_hold_s, phase_lead_steps
    del std_hold, loose_from_s, loose_until_s

    asset: Entity = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos
    assert default is not None
    offset = jump_reference_offset(
      env, self.frequency, self.n, self.crouch_hold_s, self.phase_lead_steps
    )
    reference = (default + offset)[:, self.ids]

    joint_pos = asset.data.joint_pos[:, self.ids]
    error_squared = torch.square(joint_pos - reference)
    if self.std_hold is None:
      eff_std = self.std
    else:
      t = (env.episode_length_buf.float() * env.step_dt).unsqueeze(1)
      loose = (t >= self.loose_from_s) & (t < self.loose_until_s)
      eff_std = torch.where(loose, self.std, self.std_hold)
    return torch.exp(-torch.mean(error_squared / (eff_std**2), dim=1))


class torso_clearance:
  """Graded penalty for the torso's lowest point entering the floor no-hit
  zone. Use a negative weight.

  The torso carries no collision geom, so sim lets the belly pass through the
  ground for free -- and the zuy6c85c-lineage policies loaded the jump so deep
  that the torso bottom went 2+ cm *below* the floor every episode, which on
  hardware is the battery slamming the ground and disconnecting the robot.
  The clearance is measured orientation-aware: the base body's geom bounding
  box (bottom 4.6 cm under the base origin, from the compiled model) is
  rotated by the live base quaternion, and the lowest corner is compared to
  ``min_clearance``. Returns 0 above the line, ramping linearly to 1.0 at
  floor contact and beyond 1.0 below it -- a soft barrier rather than the
  termination this started as: a hard wall priced a 1.9 cm dip like a belly
  slam and cost too much jump height, while the ramp lets the load shave into
  the zone at a small cost and makes actually touching the ground expensive.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    import mujoco
    import numpy as np

    from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import get_spec

    model = get_spec().compile()
    base = model.body("base_link").id
    corners = []
    for g in range(model.ngeom):
      if model.geom_bodyid[g] != base:
        continue
      aabb = model.geom_aabb[g]
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, model.geom_quat[g])
      c, h = aabb[:3], aabb[3:]
      pts = (
        np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        * h
        + c
      )
      corners.append((rot.reshape(3, 3) @ pts.T).T + model.geom_pos[g])
    assert corners, "base_link has no geoms to bound"
    self._corners = torch.tensor(
      np.concatenate(corners), device=env.device, dtype=torch.float32
    )  # (K, 3) in the base frame

  def __call__(
    self, env: "ManagerBasedRlEnv", min_clearance: float, asset_cfg=_ROBOT
  ) -> torch.Tensor:
    from mjlab.utils.lab_api.math import quat_apply

    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) wxyz
    n, k = quat.shape[0], self._corners.shape[0]
    world = quat_apply(
      quat.unsqueeze(1).expand(n, k, 4).reshape(-1, 4),
      self._corners.unsqueeze(0).expand(n, k, 3).reshape(-1, 3),
    ).view(n, k, 3)
    bottom = asset.data.root_link_pos_w[:, 2] + world[:, :, 2].min(dim=1).values
    return torch.clamp((min_clearance - bottom) / min_clearance, min=0.0)


def _post_launch_gate(env: "ManagerBasedRlEnv", gate_before_s: float) -> torch.Tensor:
  """1.0 once the reference launch window has opened, 0.0 during the hold.

  Both jump rewards are gated to the post-launch window: run zuy6c85c earned
  them with an airborne windup pre-hop during the hold, recycling the reset
  drop's recoil -- energy a real robot standing on the ground does not have,
  and the pre-hop is what made the hardware pre-jump pose unstable. Bouncing
  before the reference launches now earns exactly nothing.
  """
  t = env.episode_length_buf.float() * env.step_dt
  return (t >= gate_before_s).float()


def jump_up_velocity(
  env: "ManagerBasedRlEnv",
  gate_before_s: float = 0.0,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Upward base velocity, clamped at zero -- the dense launch gradient.

  An apex-shaped height reward pays nothing *locally* for pushing off the
  ground, and action noise never assembles a coherent four-leg launch on its
  own; this term pays every increment of upward push immediately. Its episode
  integral is proportional to the total rise, and the clamp means descent is
  never punished (that is the landing, not a mistake). Zero before
  ``gate_before_s`` -- see ``_post_launch_gate``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  up = torch.clamp(asset.data.root_link_lin_vel_w[:, 2], min=0.0)
  return _post_launch_gate(env, gate_before_s) * up


def jump_hold_descent(
  env: "ManagerBasedRlEnv",
  gate_after_s: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Downward base speed during the pre-launch hold. Use a negative weight.

  The complement of the hold-phase defenses: the reward gate keeps hold-phase
  motion from *earning* and the strict tracking std prices the *pose*, but a
  quick drop through tolerable poses could still bank downward momentum for
  the launch. This prices the motion itself: clamp(-vz, 0), active only
  before ``gate_after_s`` -- the countermovement dip inside the playback and
  the ballistic descent after the apex are the jump, not an exploit, and stay
  free.
  """
  asset: Entity = env.scene[asset_cfg.name]
  down = torch.clamp(-asset.data.root_link_lin_vel_w[:, 2], min=0.0)
  return (1.0 - _post_launch_gate(env, gate_after_s)) * down


def jump_airborne_height(
  env: "ManagerBasedRlEnv",
  stand_height: float,
  gate_before_s: float = 0.0,
  height_cap: float = float("inf"),
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Base height above standing, paid only while all four feet are airborne.

  Linear up to ``height_cap``, then flat: once the achievable apex is known,
  the cap makes the marginal centimeter past it worth exactly nothing, so a
  policy that has reached it spends the rest of its capacity on form (the
  tracking, symmetry and attitude terms) instead of contorting for height.
  The flight gate is what makes it unhackable -- standing tall on extended
  legs earns exactly nothing. Zero before ``gate_before_s`` -- see
  ``_post_launch_gate``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  airborne = (air_time > 0.0).all(dim=1)
  height = torch.clamp(
    asset.data.root_link_pos_w[:, 2] - stand_height, min=0.0, max=height_cap
  )
  return _post_launch_gate(env, gate_before_s) * airborne.float() * height
