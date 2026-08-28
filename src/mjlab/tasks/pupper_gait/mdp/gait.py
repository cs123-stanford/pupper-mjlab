"""Trot gait reference tracking (BeyondMimic-style reference input).

Rather than perceiving a raw phase, the policy is fed the *reference pose motion*
directly: the target joint offsets (from the default pose) it should be at, which
it then tracks. The reference is the CS 123 Pupper trot -- per-leg triangular foot
trajectories through IK, precomputed into a phase -> 12-joint-angle table (see
:mod:`gait_reference`) -- blended toward the static standing pose by command
speed:

    blend      = clamp(||command|| / blend_speed, 0, 1)
    ref_offset = blend * (trot_ref[phase] - default_pose)

So at zero command the reference offset is all zeros (stand at the default pose)
and at speed it is the full trot. Velocity itself is already in the 36-dim
proprioceptive frame, so the policy sees the reference motion and the command
together. The reward tracks this same blended reference (no separate standing
gate needed).

A stateless gait phase from each env's step counter drives the table index.
Adding the 12 reference dims makes the actor 48-dim -> 960 with history; deploying
would need the same reference computed robot-side (a follow-up).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper.mdp.observations import PupperObsNoise, pupper_actor_obs
from mjlab.tasks.pupper_gait.mdp.gait_reference import build_joint_reference_table
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_DEFAULT_NOISE = PupperObsNoise()

# |vx| below this is treated as "not translating", so the gait direction falls
# back to the turn (yaw) sign. Above it, forward/backward takes priority.
_DIR_THRESHOLD = 0.05

# Cache the clamped torch reference table per (n_samples, gait, device).
_TABLE_CACHE: dict[tuple[int, str, str], torch.Tensor] = {}


def gait_phase(env: "ManagerBasedRlEnv", frequency: float) -> torch.Tensor:
  """Per-env gait phase in [0, 1), from the episode step counter."""
  t = env.episode_length_buf.float() * env.step_dt
  return torch.frac(t * frequency)


def _joint_limits(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
  """Per-joint (lower, upper) position limits, in the entity's joint order.

  Read from the entity's own indexing rather than by name. The compiled scene
  prefixes every element with the entity name ("robot/leg_front_r_1"), so a bare
  ``mj_name2id`` lookup silently returns -1 -- and ``jnt_range[-1]`` is a valid
  row, so it returns the *last* joint's range for every joint instead of raising.
  That clamped every reference table to one arbitrary joint's limits: 0.51 rad too
  tight on the forward-drive joints, and too loose on the knees, where it would let
  the reference ask for angles the hardware cannot reach.
  """
  robot: Entity = env.scene["robot"]
  limits = robot.data.joint_pos_limits[0].to(env.device)
  return limits[:, 0].contiguous(), limits[:, 1].contiguous()


def _reference_table(
  env: "ManagerBasedRlEnv", n_samples: int, gait: str
) -> torch.Tensor:
  """Cached (n_samples, 12) joint-angle table for a gait, clamped to joint limits."""
  key = (n_samples, gait, str(env.device))
  if key not in _TABLE_CACHE:
    table = torch.tensor(
      build_joint_reference_table(n_samples, gait),
      device=env.device,
      dtype=torch.float32,
    )
    lower, upper = _joint_limits(env)
    _TABLE_CACHE[key] = torch.clamp(table, lower, upper)
  return _TABLE_CACHE[key]


def _lookup(table: torch.Tensor, eff_phase: torch.Tensor, n: int) -> torch.Tensor:
  """Linear table lookup at eff_phase in [0, 1)."""
  x = eff_phase * n
  i0 = torch.floor(x).long() % n
  i1 = (i0 + 1) % n
  frac = (x - torch.floor(x)).unsqueeze(1)
  return table[i0] * (1.0 - frac) + table[i1] * frac


def reference_offset_from_tables(
  trot_table: torch.Tensor,
  gallop_table: torch.Tensor,
  t: torch.Tensor,
  vx: torch.Tensor,
  vy: torch.Tensor,
  yaw: torch.Tensor,
  default_joint_pos: torch.Tensor,
  frequency: float,
  blend_speed: float,
  gallop_speed: float,
  gallop_freq_mult: float,
) -> torch.Tensor:
  """(N, 12) reference joint offset from the phase tables and raw command/clock.

  This is the env-free core of :func:`reference_offset`: everything the on-robot
  controller has to reproduce, given the same clamped tables. Keep the two in
  lockstep -- ``mjlab.tasks.pupper_gait.export`` ships these tables to the robot
  and mirrors this function in NumPy for parity testing.

  ``t`` is seconds since the phase clock started (episode time in sim, time since
  controller activation on the robot); only its fractional part matters.
  """
  n_samples = trot_table.shape[0]
  galloping = vx.abs() >= gallop_speed  # (N,)

  # Per-env phase clock; gallop envs run faster.
  freq = frequency * torch.where(
    galloping, torch.full_like(vx, gallop_freq_mult), torch.ones_like(vx)
  )
  phase = torch.frac(t * freq)

  dir_signal = torch.where(vx.abs() >= _DIR_THRESHOLD, vx, yaw)
  # Time-reverse when the direction is negative. Note torch.frac keeps the sign
  # (frac(-0.3) = -0.3, not 0.7); the floor-modulo in _lookup wraps it, which is
  # what makes this equivalent to indexing at 1 - phase.
  eff_phase = torch.where(dir_signal >= 0, phase, torch.frac(-phase))

  trot = _lookup(trot_table, eff_phase, n_samples)
  gallop = _lookup(gallop_table, eff_phase, n_samples)
  gait_ref = torch.where(galloping.unsqueeze(1), gallop, trot)  # (N, 12)

  offset = gait_ref - default_joint_pos

  speed = torch.linalg.vector_norm(
    torch.stack([vx, vy, yaw], dim=-1), dim=-1, keepdim=True
  )
  blend = torch.clamp(speed / blend_speed, 0.0, 1.0)
  return blend * offset


def reference_offset(
  env: "ManagerBasedRlEnv",
  command_name: str,
  frequency: float,
  blend_speed: float,
  n_samples: int,
  gallop_speed: float,
  gallop_freq_mult: float,
  phase_lead_steps: float = 0.0,
) -> torch.Tensor:
  """(N, 12) reference joint offset from default: blend * (gait_ref - default).

  The gait switches from a diagonal trot to a rotatory gallop per env once the
  commanded |vx| reaches ``gallop_speed`` (selected on the command, which is
  piecewise-constant, so it doesn't flip-flop). Galloping envs also advance the
  phase clock ``gallop_freq_mult`` x faster (a quicker cadence at high speed). For
  backward / negative-turn commands the reference is time-reversed; forward/
  backward (vx) sets the direction and the turn sign (yaw) is used only when vx is
  negligible.

  ``phase_lead_steps`` rewinds the phase clock by that many control steps. Used by
  the *reward* to close the gap between the reference the policy observed and the
  one it is graded against, so it is not required to learn a constant feedforward
  lead. Leave at 0 for the observation, which must stay reproducible on the robot.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None

  robot: Entity = env.scene["robot"]
  default = robot.data.default_joint_pos
  assert default is not None

  return reference_offset_from_tables(
    _reference_table(env, n_samples, "trot"),
    _reference_table(env, n_samples, "gallop"),
    (env.episode_length_buf.float() - phase_lead_steps) * env.step_dt,
    command[:, 0],
    command[:, 1],
    command[:, 2],
    default,
    frequency=frequency,
    blend_speed=blend_speed,
    gallop_speed=gallop_speed,
    gallop_freq_mult=gallop_freq_mult,
  )


def gait_reference_obs(
  env: "ManagerBasedRlEnv",
  command_name: str = "twist",
  frequency: float = 2.0,
  blend_speed: float = 0.1,
  n_samples: int = 100,
  gallop_speed: float = 0.5,
  gallop_freq_mult: float = 2.0,
) -> torch.Tensor:
  """Reference pose motion the policy tracks (12 = ref_offset from default)."""
  return reference_offset(
    env,
    command_name,
    frequency,
    blend_speed,
    n_samples,
    gallop_speed,
    gallop_freq_mult,
  )


def pupper_gait_actor_obs(
  env: "ManagerBasedRlEnv",
  command_name: str = "twist",
  add_noise: bool = True,
  frequency: float = 2.0,
  blend_speed: float = 0.1,
  n_samples: int = 100,
  gallop_speed: float = 0.5,
  gallop_freq_mult: float = 2.0,
  noise: PupperObsNoise = _DEFAULT_NOISE,
  asset_cfg: SceneEntityCfg = _ROBOT,
  imu_latency_distribution: Sequence[float] | None = None,
) -> torch.Tensor:
  """36-dim proprio frame (includes velocity command) + 12 reference offset = 48.

  The gait reference is not latency-randomized: the robot computes it itself from
  its own phase clock, so it is never stale (see ``gait_reference.hpp``).
  """
  base = pupper_actor_obs(
    env, command_name, add_noise, noise, asset_cfg, imu_latency_distribution
  )
  ref = reference_offset(
    env, command_name, frequency, blend_speed, n_samples, gallop_speed, gallop_freq_mult
  )
  return torch.cat([base, ref], dim=-1)


def gait_base_height(
  env: "ManagerBasedRlEnv",
  target_height: float,
  std: float,
  gallop_std: float,
  gallop_speed: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Base-height reward whose tolerance loosens to ``gallop_std`` while galloping.

  Same exp(-dz^2/std^2) height tracking as the velocity task, but when the
  commanded |vx| reaches ``gallop_speed`` the std widens (e.g. to 10) so height is
  effectively unconstrained at top speed, letting the body pitch/bob in the gallop.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  galloping = command[:, 0].abs() >= gallop_speed
  eff_std = torch.where(
    galloping,
    torch.full_like(height, gallop_std),
    torch.full_like(height, std),
  )
  return torch.exp(-torch.square(height - target_height) / eff_std**2)


class gait_tracking:
  """Reward for matching the blended reference pose, exp(-mean(error^2/std^2)).

  Tracks ``default + reference_offset`` -- i.e. the static pose at zero command
  and the trot at speed -- so no separate standing gate is needed.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.ids = cfg.params["asset_cfg"].joint_ids
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"], list_of_strings=joint_names
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)
    self.command_name = cfg.params["command_name"]
    self.frequency = float(cfg.params["frequency"])
    self.blend_speed = float(cfg.params["blend_speed"])
    self.n = int(cfg.params["n_samples"])
    self.gallop_speed = float(cfg.params["gallop_speed"])
    self.gallop_freq_mult = float(cfg.params["gallop_freq_mult"])
    self.phase_lead_steps = float(cfg.params.get("phase_lead_steps", 0.0))
    _reference_table(env, self.n, "trot")  # warm the caches at construction
    _reference_table(env, self.n, "gallop")

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    std,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    frequency: float,
    blend_speed: float,
    n_samples: int,
    gallop_speed: float,
    gallop_freq_mult: float,
    phase_lead_steps: float = 0.0,
  ) -> torch.Tensor:
    del std, command_name, frequency, blend_speed, n_samples
    del gallop_speed, gallop_freq_mult, phase_lead_steps

    asset: Entity = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos
    assert default is not None
    offset = reference_offset(
      env,
      self.command_name,
      self.frequency,
      self.blend_speed,
      self.n,
      self.gallop_speed,
      self.gallop_freq_mult,
      self.phase_lead_steps,
    )
    reference = (default + offset)[:, self.ids]

    joint_pos = asset.data.joint_pos[:, self.ids]
    error_squared = torch.square(joint_pos - reference)
    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


# Shared actor/critic observation.
#
# ``pupperv3-mjx`` has no privileged critic at all: ``environment._get_obs``
# returns a single flat array that ``state.replace(obs=...)`` hands to both
# networks, and ``make_ppo_networks`` is called without a ``value_obs_key``. That
# array is fully corrupted -- noise on every subgroup and the IMU latency buffer
# applied -- so the critic sees exactly the actor's degraded view.
#
# Reproducing that here needs care. The observation manager evaluates each group
# separately within one ``compute()``, and the IMU latency buffer lives on the env
# and is shared, so pointing both groups at ``pupper_gait_actor_obs`` would advance
# that buffer twice per env step -- halving the effective lag and handing the two
# groups different IMU samples for the same instant. The frame is therefore
# computed once per step and cached.
_SHARED_FRAME_ATTR = "_pupper_gait_shared_frame"


def pupper_gait_shared_obs(env: "ManagerBasedRlEnv", **kwargs) -> torch.Tensor:
  """The actor frame, computed once per env step and reused by every group.

  Use for both the actor and the critic when they share an observation. See the
  module note above for why this cannot simply call
  :func:`pupper_gait_actor_obs` twice.
  """
  cached = getattr(env, _SHARED_FRAME_ATTR, None)
  if cached is not None and cached[0] == env.common_step_counter:
    return cached[1]
  obs = pupper_gait_actor_obs(env, **kwargs)
  setattr(env, _SHARED_FRAME_ATTR, (env.common_step_counter, obs))
  return obs


def reset_gait_shared_obs(
  env: "ManagerBasedRlEnv", env_ids: torch.Tensor | slice | None
) -> None:
  """Event term: drop the cached frame so a reset cannot serve a stale one.

  ``common_step_counter`` does not advance on ``reset()``, so back-to-back resets
  would otherwise hit the cache and return the pre-reset observation.
  """
  del env_ids  # The frame is computed for all envs at once.
  if hasattr(env, _SHARED_FRAME_ATTR):
    delattr(env, _SHARED_FRAME_ATTR)
