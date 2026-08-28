"""MixedGaits with an insertable one-shot jump slot.

The per-command mixed reference (trot / lift / captured fast gaits) with a jump
that can be requested at any time -- stationary or mid-locomotion -- and takes
effect at the next *gait gap*: the base cycle is 0.75 s and the fast gaits run
on a 2.5x clock (0.3 s), so every ``2 / frequency`` seconds (1.5 s) is a shared
phase boundary of every gait. Jump slots start on that grid and last exactly
one grid unit: the captured jump plays over its real duration, then the landing
pose (~ the capture's own near-stance row 0) holds for the remainder, and the
locomotion clock -- which keeps running underneath -- exits the slot exactly at
its own boundary. No clock surgery, and the same arithmetic ports to the
robot's wall-clock phase.

Inside a slot the reference is full strength (the command-norm blend that
zeroes the locomotion reference at zero command is bypassed -- a stationary
jump is still a jump), and the commanded twist stays live in the observation
and the velocity rewards: that is the signal for carrying the commanded
velocity through the air.

The schedule is decided at reset (``reset_jump_schedule``) and stored as a
per-env tensor of slot-start times on the env; every consumer derives
"in slot" purely from ``episode_length_buf`` against that tensor, so the
tracking reward's phase-lead rewind (which temporarily mutates the clock)
shifts the slot window consistently for free -- and the trigger is invisible
to the policy until the reference swaps, so pre-jump anticipation cannot be
learned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.pupper.mdp.observations import PupperObsNoise, pupper_actor_obs
from mjlab.tasks.pupper.mdp.velocity_command import (
  PupperVelocityCommand,
  PupperVelocityCommandCfg,
)
from mjlab.tasks.pupper_gait.mdp.gait import _lookup, _reference_table
from mjlab.tasks.pupper_gait.mdp.mixed_gaits import mixed_reference_offset
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_DEFAULT_NOISE = PupperObsNoise()

# Per-env slot-start times [s], (num_envs, _MAX_JUMPS), +inf = unused slot.
_JUMP_SCHEDULE_ATTR = "_pupper_mixed_jump_slots"
_MAX_JUMPS = 3


def _slot_starts(env: "ManagerBasedRlEnv") -> torch.Tensor:
  buf = getattr(env, _JUMP_SCHEDULE_ATTR, None)
  if buf is None:
    buf = torch.full((env.num_envs, _MAX_JUMPS), float("inf"), device=env.device)
    setattr(env, _JUMP_SCHEDULE_ATTR, buf)
  return buf


def reset_jump_schedule(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  grid_s: float,
  p_no_jump: float = 0.25,
  min_start_s: float = 1.5,
  end_margin_s: float = 2.5,
  min_separation_s: float = 3.0,
) -> None:
  """Reset event: sample this episode's jump-slot start times.

  Per env: no jump with probability ``p_no_jump`` (pure locomotion stays in
  distribution), else one to three slots, uniform in the episode and
  quantized *up* to the ``grid_s`` gait-gap grid -- where a real trigger
  would snap to. ``min_separation_s`` keeps slots disjoint even after
  quantization.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  buf = _slot_starts(env)
  n = len(env_ids)
  # Play configs set the episode to ~forever; cap the scheduling horizon so a
  # viewer session still sees its jumps within the first half minute.
  episode_s = min(env.max_episode_length * env.step_dt, 30.0)

  def _quantize(t: torch.Tensor) -> torch.Tensor:
    return torch.ceil(t / grid_s) * grid_s

  starts = torch.full((n, _MAX_JUMPS), float("inf"), device=env.device)
  r = torch.rand(n, device=env.device)
  # 0 with p_no_jump, else 1..3 uniformly.
  count = torch.where(
    r < p_no_jump,
    torch.zeros_like(r),
    1.0 + torch.floor(3.0 * (r - p_no_jump) / (1.0 - p_no_jump)).clamp(max=2.0),
  )

  hi = episode_s - end_margin_s
  t1 = _quantize(
    torch.empty(n, device=env.device).uniform_(min_start_s, max(hi, min_start_s + 1e-3))
  )
  prev = t1
  cols = [t1]
  for _ in range(_MAX_JUMPS - 1):
    lo = prev + min_separation_s
    span = (hi - lo).clamp(min=0.0)
    tk = _quantize(lo + span * torch.rand(n, device=env.device))
    cols.append(tk)
    prev = tk

  inf = torch.full_like(t1, float("inf"))
  for k, tk in enumerate(cols):
    fits = tk <= hi + grid_s
    starts[:, k] = torch.where((count >= k + 1) & fits, tk, inf)
  buf[env_ids] = starts


def request_jump(
  env: "ManagerBasedRlEnv", env_idx: int, grid_s: float, busy_s: float | None = None
) -> float:
  """Schedule a jump slot at the next gait gap for one env; returns its start.

  The interactive analogue of the training schedule -- the play viewer's Jump
  button (and eventually the robot's X button) calls this. The slot starts at
  the next grid point strictly ahead of now, pushed past any pending or
  still-active window (``busy_s``, default one grid unit): with the grid finer
  than the reward window, mashing the button must queue the next jump after
  the current one instead of restarting it mid-flight. Scheduling the same
  start twice is a no-op.
  """
  import math

  busy = grid_s if busy_s is None else busy_s
  buf = _slot_starts(env)
  t = float(env.episode_length_buf[env_idx]) * env.step_dt
  start = math.ceil((t + 1e-6) / grid_s) * grid_s
  row = buf[env_idx]
  for _ in range(_MAX_JUMPS + 1):
    blocked = False
    for r in row.tolist():
      if r != float("inf") and r <= start < r + busy:
        start = math.ceil((r + busy - 1e-6) / grid_s) * grid_s
        blocked = True
    if not blocked:
      break
  if bool(((row - start).abs() < 1e-6).any()):
    return start  # already scheduled for that gap
  free = torch.isinf(row) | (row + busy <= t)
  col = int(free.nonzero()[0]) if bool(free.any()) else int(row.argmax())
  buf[env_idx, col] = start
  return start


class MixedJumpVelocityCommand(PupperVelocityCommand):
  """The Pupper velocity command plus a Jump button in the play viewer."""

  cfg: "MixedJumpVelocityCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
    super().create_gui(name, server, get_env_idx, on_change, request_action)
    grid_s = self.cfg.jump_slot_s
    busy_s = self.cfg.jump_busy_s
    button = server.gui.add_button("Jump")

    @button.on_click
    def _(_) -> None:
      start = request_jump(self._env, get_env_idx(), grid_s, busy_s)
      t = float(self._env.episode_length_buf[get_env_idx()]) * self._env.step_dt
      print(f"[viewer] jump scheduled at t={start:.2f}s (in {start - t:.2f}s)")


@dataclass
class MixedJumpVelocityCommandCfg(PupperVelocityCommandCfg):
  """Pupper velocity command cfg whose term offers the viewer Jump button."""

  jump_slot_s: float = 0.75
  jump_busy_s: float = 1.0

  def build(self, env: "ManagerBasedRlEnv") -> MixedJumpVelocityCommand:
    return MixedJumpVelocityCommand(self, env)


def _slot_progress(
  env: "ManagerBasedRlEnv", slot_s: float
) -> tuple[torch.Tensor, torch.Tensor]:
  """(in_slot (N,) bool, seconds-into-slot (N,), 0 where not in a slot).

  Pure function of ``episode_length_buf`` and the reset-time schedule, so the
  tracking reward's phase-lead rewind shifts the window consistently.
  """
  buf = _slot_starts(env)
  t = (env.episode_length_buf.float() * env.step_dt).unsqueeze(1)  # (N, 1)
  d = t - buf  # (N, K)
  in_each = (d >= 0.0) & (d < slot_s)
  in_slot = in_each.any(dim=1)
  d_masked = torch.where(in_each, d, torch.full_like(d, float("inf")))
  t_in = d_masked.min(dim=1).values
  return in_slot, torch.where(in_slot, t_in, torch.zeros_like(t_in))


def _outside_gate(env: "ManagerBasedRlEnv", slot_s: float) -> torch.Tensor:
  in_slot, _ = _slot_progress(env, slot_s)
  return 1.0 - in_slot.float()


def mixed_jump_reference_offset(
  env: "ManagerBasedRlEnv",
  command_name: str,
  frequency: float,
  blend_speed: float,
  n_samples: int,
  gallop_speed: float,
  jump_duration_s: float,
  slot_s: float,
  cross_fade_s: float = 0.06,
) -> torch.Tensor:
  """(N, 12) composite reference offset: the mixed gaits, or the jump slot.

  Outside a slot this is byte-identical to :func:`mixed_reference_offset`.
  Inside, the captured jump table plays once over ``jump_duration_s`` (phase
  clamped at 1: the loop-closed landing pose holds for the slot remainder) at
  full strength -- no command blend, a stationary jump is still a jump.

  A ``cross_fade_s`` linear fade at each slot edge makes the composite
  continuous by construction: the fade weight is 0 exactly at the boundaries,
  so entering blends locomotion-touchdown -> jump get-down and exiting blends
  landing crouch -> locomotion touchdown, cutting the raw boundary snap
  (0.2-0.5 rad in one control step, worst across command modes) to per-step
  changes the actuators track anyway. Stateless, so it ports to the robot's
  wall-clock unchanged.
  """
  base = mixed_reference_offset(
    env, command_name, frequency, blend_speed, n_samples, gallop_speed
  )
  in_slot, t_in = _slot_progress(env, slot_s)
  if not bool(in_slot.any()):
    return base

  robot: Entity = env.scene["robot"]
  default = robot.data.default_joint_pos
  assert default is not None
  phase = torch.clamp(t_in / jump_duration_s, min=0.0, max=1.0)
  jump_ref = _lookup(_reference_table(env, n_samples, "jump"), phase, n_samples)
  fade = torch.clamp(
    torch.minimum(t_in, slot_s - t_in) / cross_fade_s, min=0.0, max=1.0
  ).unsqueeze(1)
  slot_ref = fade * (jump_ref - default) + (1.0 - fade) * base
  return torch.where(in_slot.unsqueeze(1), slot_ref, base)


def mixed_jump_reference_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  frequency: float,
  blend_speed: float,
  n_samples: int,
  gallop_speed: float,
  jump_duration_s: float,
  slot_s: float,
  cross_fade_s: float = 0.06,
) -> torch.Tensor:
  """Composite reference offset for the critic (12 dims)."""
  return mixed_jump_reference_offset(
    env,
    command_name,
    frequency,
    blend_speed,
    n_samples,
    gallop_speed,
    jump_duration_s,
    slot_s,
    cross_fade_s,
  )


def mixed_jump_actor_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  frequency: float,
  blend_speed: float,
  n_samples: int,
  gallop_speed: float,
  jump_duration_s: float,
  slot_s: float,
  cross_fade_s: float = 0.06,
  add_noise: bool = True,
  noise: PupperObsNoise = _DEFAULT_NOISE,
  asset_cfg: SceneEntityCfg = _ROBOT,
  imu_latency_distribution: Sequence[float] | None = None,
) -> torch.Tensor:
  """36-dim proprio frame (live twist command included) + 12 composite = 48.

  The reference dims are how the policy observes the jump regime -- the slot
  is visible the moment it starts and not an instant before.
  """
  base = pupper_actor_obs(
    env, command_name, add_noise, noise, asset_cfg, imu_latency_distribution
  )
  ref = mixed_jump_reference_offset(
    env,
    command_name,
    frequency,
    blend_speed,
    n_samples,
    gallop_speed,
    jump_duration_s,
    slot_s,
    cross_fade_s,
  )
  return torch.cat([base, ref], dim=-1)


class mixed_jump_tracking:
  """Track the composite reference, exp(-mean(error^2/std^2)).

  One term over both regimes -- the reference swap does the regime switch, so
  there is nothing for two tracking terms to disagree about. The tolerance is
  the locomotion ``std`` outside a slot and the looser ``std_slot`` inside it
  (the jump is a template the policy may out-jump; the height terms drive the
  apex), the same phase-dependent-std treatment the standalone jump task uses.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.ids = cfg.params["asset_cfg"].joint_ids
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    def _resolve(spec) -> torch.Tensor:
      _, _, vals = resolve_matching_names_values(data=spec, list_of_strings=joint_names)
      return torch.tensor(vals, device=env.device, dtype=torch.float32)

    self.std = _resolve(cfg.params["std"])
    self.std_slot = _resolve(cfg.params["std_slot"])
    self._p = {
      k: cfg.params[k]
      for k in (
        "command_name",
        "frequency",
        "blend_speed",
        "n_samples",
        "gallop_speed",
        "jump_duration_s",
        "slot_s",
        "cross_fade_s",
      )
    }
    # The loose-std window outlives the reference window: the reference
    # stitches to the gait mid-descent, but the robot is still airborne and
    # cannot track a stride tightly until it lands.
    self.std_loose_s = float(cfg.params["std_loose_s"])
    self.phase_lead_steps = float(cfg.params["phase_lead_steps"])
    for gait in ("trot", "reach", "reach_back", "lift", "jump"):
      _reference_table(env, int(self._p["n_samples"]), gait)

  def __call__(self, env: "ManagerBasedRlEnv", **kwargs) -> torch.Tensor:
    del kwargs
    asset: Entity = env.scene["robot"]
    default = asset.data.default_joint_pos
    assert default is not None

    # Rewind the clock for the reward only (see GAIT_REWARD_PHASE_LEAD_STEPS);
    # the slot window is derived from the same clock, so it shifts with it.
    saved = env.episode_length_buf
    env.episode_length_buf = (saved.float() - self.phase_lead_steps).to(saved.dtype)
    try:
      offset = mixed_jump_reference_offset(env, **self._p)
    finally:
      env.episode_length_buf = saved
    reference = (default + offset)[:, self.ids]

    in_loose, _ = _slot_progress(env, self.std_loose_s)
    eff_std = torch.where(in_loose.unsqueeze(1), self.std_slot, self.std)
    joint_pos = asset.data.joint_pos[:, self.ids]
    error_squared = torch.square(joint_pos - reference)
    return torch.exp(-torch.mean(error_squared / (eff_std**2), dim=1))


# --- Slot-gated height ladder (the jump task's terms, keyed on the slot). ---


def slot_jump_up_velocity(
  env: "ManagerBasedRlEnv",
  slot_s: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Upward base velocity, clamped at zero, paid only inside a jump slot."""
  in_slot, _ = _slot_progress(env, slot_s)
  asset: Entity = env.scene[asset_cfg.name]
  up = torch.clamp(asset.data.root_link_lin_vel_w[:, 2], min=0.0)
  return in_slot.float() * up


def slot_jump_airborne_height(
  env: "ManagerBasedRlEnv",
  slot_s: float,
  stand_height: float,
  height_cap: float = float("inf"),
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Capped flight-gated height above standing, paid only inside a jump slot."""
  in_slot, _ = _slot_progress(env, slot_s)
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  airborne = (air_time > 0.0).all(dim=1)
  height = torch.clamp(
    asset.data.root_link_pos_w[:, 2] - stand_height, min=0.0, max=height_cap
  )
  return in_slot.float() * airborne.float() * height


# --- Window-aware velocity tracking (the carry-through surgery). ---


def track_linear_velocity_windowed(
  env: "ManagerBasedRlEnv",
  std: float,
  command_name: str,
  slot_s: float,
  slot_weight_mult: float = 1.0,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Stock linear-velocity tracking outside a jump slot, xy-only inside.

  The shared term folds ``vz^2`` into the error ("commanded z assumed zero"),
  which floors it to ~0 in flight -- no xy gradient exactly when carrying the
  commanded velocity through the air must be learned, and a tax on jumping at
  all. Dropping the z term only inside the slot keeps locomotion bit-identical
  to the stock term and makes flight xy-velocity the thing the slot pays for.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  outside = _outside_gate(env, slot_s)
  # In-window boost: carrying the commanded velocity through the jump is the
  # slot's headline objective, so it earns more there than plain locomotion
  # tracking does (per-step multiplier = an effective in-window weight raise).
  boost = 1.0 + (slot_weight_mult - 1.0) * (1.0 - outside)
  return boost * torch.exp(-(xy_error + outside * z_error) / std**2)


# --- Off-in-slot wrappers for shared terms that fight a jump. ---
#
# Each is the stock term multiplied per step by the outside-slot gate: inside a
# slot the term contributes exactly zero reward and zero gradient. Explicit
# thin wrappers (rather than one generic decorator) because the reward manager
# resolves params against the function signature.


def base_height_outside_slot(env: "ManagerBasedRlEnv", slot_s: float, **kwargs):
  from mjlab.tasks.pupper_gait.mdp.gait import gait_base_height

  return _outside_gate(env, slot_s) * gait_base_height(env, **kwargs)


def lin_vel_z_l2_outside_slot(env: "ManagerBasedRlEnv", slot_s: float, **kwargs):
  from mjlab.tasks.pupper.mdp import lin_vel_z_l2

  return _outside_gate(env, slot_s) * lin_vel_z_l2(env, **kwargs)


def air_time_outside_slot(env: "ManagerBasedRlEnv", slot_s: float, **kwargs):
  from mjlab.tasks.velocity.mdp import feet_air_time

  return _outside_gate(env, slot_s) * feet_air_time(env, **kwargs)


def foot_slip_outside_slot(env: "ManagerBasedRlEnv", slot_s: float, **kwargs):
  from mjlab.tasks.velocity.mdp import feet_slip

  return _outside_gate(env, slot_s) * feet_slip(env, **kwargs)


def stand_still_pose_outside_slot(env: "ManagerBasedRlEnv", slot_s: float, **kwargs):
  from mjlab.tasks.pupper.mdp import stand_still_pose

  return _outside_gate(env, slot_s) * stand_still_pose(env, **kwargs)


def stand_still_joint_velocity_outside_slot(
  env: "ManagerBasedRlEnv", slot_s: float, **kwargs
):
  from mjlab.tasks.pupper.mdp import stand_still_joint_velocity

  return _outside_gate(env, slot_s) * stand_still_joint_velocity(env, **kwargs)
