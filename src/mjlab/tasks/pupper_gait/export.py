"""Export a trained Pupper *gait* policy to the on-robot ``neural_controller`` JSON.

The gait policy's observation frame is the velocity task's 36-dim frame plus the
12-dim reference joint offset it tracks (see :mod:`..mdp.gait`), so the deploy
frame is 48-dim and the robot has to *produce* those last 12 numbers itself every
control step.

Porting the CS 123 gradient-descent IK to C++ would be a needless liability, so
this exporter ships the already-computed, joint-limit-clamped phase tables (trot
and gallop for the single-gait tasks; trot, reach-as-fast, reach_back and lift
for the mixed-reference ones) plus the blend/gallop parameters in a
``gait_reference`` block of the deploy JSON. The robot then only has to do a table lookup and a
blend -- ``neural_controller``'s ``gait_reference.hpp`` is a direct
transcription of :func:`reference_offset_numpy` below, which in turn mirrors the
training-time :func:`~mjlab.tasks.pupper_gait.mdp.gait.reference_offset_from_tables`
(and, with a ``lift_table``,
:func:`~mjlab.tasks.pupper_gait.mdp.mixed_gaits.mixed_reference_offset`).

Layout of the exported 48-dim frame (robot's newest-first history)::

    [ body_ang_vel(3), projected_gravity(3), command(3), desired_world_z(3),
      joint_pos - default(12), last_action(12), gait_reference_offset(12) ]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  SINGLE_OBS_DIM,
)
from mjlab.tasks.pupper.export import (
  BASE_OBS_COMPONENTS,
  export_pupper_policy_from_env,
)
from mjlab.tasks.pupper_gait.mdp.gait import (
  _DIR_THRESHOLD,
  _joint_limits,
  _reference_table,
)
from mjlab.tasks.pupper_gait.mdp.gait_reference import _GAIT_FREQ_MULT

# The gait actor frame: 36-dim proprio + the 12-dim reference offset it tracks.
GAIT_SINGLE_OBS_DIM: int = SINGLE_OBS_DIM + len(JOINT_NAMES)

# The gait frame layout for the deploy obs_spec: the base menu plus the
# robot-computed reference offset.
GAIT_OBS_COMPONENTS: tuple[tuple[str, int], ...] = BASE_OBS_COMPONENTS + (
  ("gait_reference", len(JOINT_NAMES)),
)

# Index of the reference-offset block within a frame (must match the on-robot
# ``kGaitReferenceIdx``).
GAIT_REFERENCE_IDX: int = SINGLE_OBS_DIM

_GAIT_PARAM_KEYS = (
  "frequency",
  "blend_speed",
  "n_samples",
  "gallop_speed",
  "gallop_freq_mult",
)

# The mixed-reference tasks (MixedGaits, StableGait) thread the same phase-clock
# parameters minus the freq multiplier, which for them lives in the reference
# module (the reach gait's _GAIT_FREQ_MULT entry).
_MIXED_GAIT_PARAM_KEYS = (
  "frequency",
  "blend_speed",
  "n_samples",
  "gallop_speed",
)

# The jump task's one-shot reference (see ..mdp.jump): no blend, no command
# coupling -- just a clock. crouch_hold_s is unique to it, so its presence is
# the frame-kind discriminator.
_JUMP_PARAM_KEYS = (
  "frequency",
  "n_samples",
  "crouch_hold_s",
)

# MixedGaitsJump: the mixed keys plus the jump-slot clock. jump_duration_s is
# unique to it and is the discriminator.
_MIXED_JUMP_PARAM_KEYS = (
  "frequency",
  "blend_speed",
  "n_samples",
  "gallop_speed",
  "jump_duration_s",
  "slot_s",
  "cross_fade_s",
)


def gait_frame_kind(env) -> str | None:
  """Which 48-dim gait family the actor frame belongs to, or None for 36-dim.

  ``"single"`` is the trot/gallop family (:mod:`..mdp.gait`), whose fast branch
  plays the gallop table; ``"mixed"`` is the per-command family
  (:mod:`..mdp.mixed_gaits`: MixedGaits, StableGait), which adds a lift-in-place
  table for turning/sidestepping and whose fast branch plays the reach gait;
  ``"jump"`` is the single-jump task (:mod:`..mdp.jump`), whose reference is a
  one-shot clock rather than a command-blended cycle; ``"zeroref"`` is the
  plain velocity task in the gait frame (:func:`..mdp.zero_reference_actor_obs`),
  whose reference dims are identically zero.
  """
  params = env.observation_manager.get_term_cfg("actor", "policy").params
  if "zero_reference" in params:
    return "zeroref"
  if all(k in params for k in _JUMP_PARAM_KEYS):
    return "jump"
  if all(k in params for k in _MIXED_JUMP_PARAM_KEYS):
    return "mixed_jump"
  if all(k in params for k in _GAIT_PARAM_KEYS):
    return "single"
  if all(k in params for k in _MIXED_GAIT_PARAM_KEYS):
    return "mixed"
  return None


def gait_params_from_env(env) -> dict[str, float]:
  """Read the gait parameters off the live env's actor observation term.

  Taking them from the observation manager (rather than re-importing the task
  config defaults) keeps the exported metadata consistent with whatever the run
  was actually trained with. For mixed-reference envs, ``gallop_freq_mult`` is
  filled from the reach gait's ``_GAIT_FREQ_MULT`` entry -- the training-time
  clock multiplier of the fast branch, which deploy applies to its fast table.
  """
  kind = gait_frame_kind(env)
  if kind is None:
    raise ValueError(
      "Actor observation term is missing the gait phase-clock parameters; is "
      "this a gait task env?"
    )
  term_cfg = env.observation_manager.get_term_cfg("actor", "policy")
  keys = {
    "single": _GAIT_PARAM_KEYS,
    "mixed": _MIXED_GAIT_PARAM_KEYS,
    "mixed_jump": _MIXED_JUMP_PARAM_KEYS,
    "jump": _JUMP_PARAM_KEYS,
    "zeroref": _MIXED_GAIT_PARAM_KEYS,
  }[kind]
  params = {k: term_cfg.params[k] for k in keys}
  if kind in ("mixed", "mixed_jump"):
    params["gallop_freq_mult"] = float(_GAIT_FREQ_MULT.get("reach", 1.0))
  elif kind == "zeroref":
    # Constant tables make the clock inert; the multiplier is a formality.
    params["gallop_freq_mult"] = 1.0
  return params


def gait_reference_metadata(env) -> dict[str, Any]:
  """The ``gait_reference`` deploy block: phase tables + blend/gallop parameters.

  Tables are ``(n_samples, 12)`` joint angles in ``JOINT_NAMES`` order, clamped to
  the model's joint limits exactly as training did.

  For mixed-reference envs the block reuses the single-gait schema -- the
  fast-forward slot is keyed ``gallop_table`` -- so the reach table ships under
  that key (with the reach clock multiplier as ``gallop_freq_mult``), the
  lift-in-place table is added as ``lift_table`` (played when the command is
  not translating), and the backward capture as ``gallop_back_table`` (played
  instead of the time-reversed forward table when galloping backward,
  mirroring :func:`~mjlab.tasks.pupper_gait.mdp.mixed_gaits.mixed_reference_offset`).
  A StableGait policy ships its (huge, never-crossed) ``gallop_speed``, so its
  fast tables are inert on the robot exactly as they were in training.
  """
  kind = gait_frame_kind(env)
  params = gait_params_from_env(env)
  n_samples = int(params["n_samples"])
  if kind == "zeroref":
    # An empty reference: every table row is the default pose, so the on-robot
    # offset (table minus default) is identically zero at any phase or
    # command -- exactly the zeros the policy observed in training.
    default = env.scene["robot"].data.default_joint_pos[0]
    rows = default.unsqueeze(0).expand(n_samples, -1)
    trot = rows.detach().cpu().numpy()
    fast = trot
  else:
    fast_gait = "gallop" if kind == "single" else "reach"
    trot = _reference_table(env, n_samples, "trot").detach().cpu().numpy()
    fast = _reference_table(env, n_samples, fast_gait).detach().cpu().numpy()
  # Recorded for auditability: the tables are shipped post-clamp, so the robot
  # tracks exactly the reference the policy was trained against.
  clamp_lower, clamp_upper = _joint_limits(env)
  out = {
    "clamp_lower": clamp_lower.detach().cpu().numpy().tolist(),
    "clamp_upper": clamp_upper.detach().cpu().numpy().tolist(),
    "n_samples": n_samples,
    "frequency": float(params["frequency"]),
    "blend_speed": float(params["blend_speed"]),
    "gallop_speed": float(params["gallop_speed"]),
    "gallop_freq_mult": float(params["gallop_freq_mult"]),
    "dir_threshold": float(_DIR_THRESHOLD),
    "reference_index": GAIT_REFERENCE_IDX,
    "joint_names": list(JOINT_NAMES),
    "trot_table": trot.tolist(),
    "gallop_table": fast.tolist(),
  }
  if kind in ("mixed", "mixed_jump"):
    lift = _reference_table(env, n_samples, "lift").detach().cpu().numpy()
    out["lift_table"] = lift.tolist()
    # The backward-fast reference is its own capture, not the forward one
    # time-reversed. It is shipped exactly as the training pipeline stores it
    # (pre-reversed -- see gait_reference's reach_back handling), so the
    # controller's shared backward phase reversal plays the capture forward as
    # recorded, byte-identical to training.
    back = _reference_table(env, n_samples, "reach_back").detach().cpu().numpy()
    out["gallop_back_table"] = back.tolist()
  return out


def jump_reference_metadata(env) -> dict[str, Any]:
  """The ``jump_reference`` deploy block: the one-shot jump table and its clock.

  Unlike ``gait_reference`` this carries no blend or command coupling: the
  reference is a function of time-since-trigger alone. Idle (untriggered) is
  ``t = 0``, which the clamp maps to the mid-stance crouch at ``phase_start``;
  a trigger plays exactly one cycle and lands back on the same pose (one full
  wrap). ``trigger_button`` is the joy-message button index the controller
  arms the clock on -- R2 under the standard ``joy`` PS4/PS5 mapping.
  """
  from mjlab.tasks.pupper_gait.mdp.jump import _JUMP_PHASE_START

  if gait_frame_kind(env) != "jump":
    raise ValueError("Not a jump task env.")
  params = gait_params_from_env(env)
  n_samples = int(params["n_samples"])
  table = _reference_table(env, n_samples, "jump").detach().cpu().numpy()
  clamp_lower, clamp_upper = _joint_limits(env)
  return {
    "clamp_lower": clamp_lower.detach().cpu().numpy().tolist(),
    "clamp_upper": clamp_upper.detach().cpu().numpy().tolist(),
    "n_samples": n_samples,
    "frequency": float(params["frequency"]),
    "crouch_hold_s": float(params["crouch_hold_s"]),
    "phase_start": float(_JUMP_PHASE_START),
    "trigger_button": 7,  # R2 in the standard joy PS4/PS5 mapping.
    "reference_index": GAIT_REFERENCE_IDX,
    "joint_names": list(JOINT_NAMES),
    "jump_table": table.tolist(),
  }


def jump_slot_metadata(env) -> dict[str, Any]:
  """The ``jump_slot`` deploy block for MixedGaitsJump policies.

  Everything the controller needs to insert a one-shot jump into the mixed
  reference on a button press: the (joint-limit-clamped) jump table, the
  playback clock, the active-window and cross-fade geometry, and the start
  grid the trigger quantizes to. The reference math the robot must reproduce
  is :func:`mixed_jump_reference_offset_numpy` below -- keep the C++ in step.
  """
  from mjlab.tasks.pupper_gait.mystery_env_cfg import (
    JUMP_ACTIVE_S,
    JUMP_CROSS_FADE_S,
    JUMP_PLAYBACK_S,
    JUMP_REWARD_WINDOW_S,
    JUMP_SLOT_S,
  )

  if gait_frame_kind(env) != "mixed_jump":
    raise ValueError("Not a MixedGaitsJump env.")
  params = gait_params_from_env(env)
  n_samples = int(params["n_samples"])
  table = _reference_table(env, n_samples, "jump").detach().cpu().numpy()
  # Cross-check the env's threaded values against the task constants: a
  # mismatch means the policy trained on a different slot geometry than the
  # block would ship.
  assert abs(float(params["jump_duration_s"]) - JUMP_PLAYBACK_S) < 1e-9
  assert abs(float(params["slot_s"]) - JUMP_ACTIVE_S) < 1e-9
  assert abs(float(params["cross_fade_s"]) - JUMP_CROSS_FADE_S) < 1e-9
  return {
    "n_samples": n_samples,
    "playback_s": float(JUMP_PLAYBACK_S),
    "active_s": float(JUMP_ACTIVE_S),
    "cross_fade_s": float(JUMP_CROSS_FADE_S),
    "grid_s": float(JUMP_SLOT_S),
    "busy_s": float(JUMP_REWARD_WINDOW_S),
    # Game-mode controls (deploy-side choices, shipped in the JSON so they are
    # tweakable on the robot without a rebuild): x jumps, holding circle lifts
    # the walk cap to the run cap. The walk cap sits just under the 0.5 m/s
    # fast-gait onset, so without the run button the fast references never
    # engage.
    "trigger_button": 0,  # x
    "run_button": 1,  # circle, held
    "walk_speed_cap": 0.49,
    "run_speed_cap": 1.5,
    "jump_table": table.tolist(),
  }


def heading_hold_metadata(env) -> dict[str, float] | None:
  """The ``heading_hold`` deploy block, or None for runs trained without it.

  The gait tasks close a loop through the yaw command (see
  ``UniformVelocityCommandCfg.heading_hold_kp``): walking with a quiet
  commanded yaw, the emitted yaw command is a clipped P-correction toward the
  heading captured at the quiet edge. The robot must run the identical
  correction off its IMU yaw, so the numbers ship with the export rather than
  being kept in sync by hand.
  """
  twist = env.cfg.commands.get("twist")
  kp = getattr(twist, "heading_hold_kp", None)
  if kp is None:
    return None
  return {
    "kp": float(kp),
    "clip": float(twist.heading_hold_clip),
    "yaw_threshold": float(twist.heading_hold_yaw_threshold),
    "walk_threshold": float(twist.heading_hold_walk_threshold),
  }


def export_pupper_gait_policy_from_env(
  actor: nn.Module, env, *, use_imu: bool = True
) -> dict[str, Any]:
  """Export the gait policy: the standard deploy JSON plus ``gait_reference``."""
  policy = export_pupper_policy_from_env(
    actor,
    env,
    use_imu=use_imu,
    single_obs_dim=GAIT_SINGLE_OBS_DIM,
    obs_components=GAIT_OBS_COMPONENTS,
  )
  policy["gait_reference"] = gait_reference_metadata(env)
  heading_hold = heading_hold_metadata(env)
  if heading_hold is not None:
    policy["heading_hold"] = heading_hold
  return policy


def reference_offset_numpy(
  gait: dict[str, Any],
  default_joint_pos: np.ndarray,
  t: np.ndarray,
  vx: np.ndarray,
  vy: np.ndarray,
  yaw: np.ndarray,
) -> np.ndarray:
  """NumPy mirror of the on-robot reference-offset computation.

  This is the reference implementation the C++ ``compute_gait_reference_offset``
  transcribes, and the thing parity-tested against training-time torch. ``t`` is
  seconds since the phase clock started; ``gait`` is the exported
  ``gait_reference`` block. A ``lift_table`` in the block (mixed-reference
  policies) is played instead of the trot when the command is not translating,
  and a ``gallop_back_table`` is played instead of the (time-reversed) fast
  table when galloping backward, exactly as the controller does.
  """
  trot = np.asarray(gait["trot_table"], dtype=np.float64)
  gallop = np.asarray(gait["gallop_table"], dtype=np.float64)
  lift = (
    np.asarray(gait["lift_table"], dtype=np.float64) if "lift_table" in gait else None
  )
  gallop_back = (
    np.asarray(gait["gallop_back_table"], dtype=np.float64)
    if "gallop_back_table" in gait
    else None
  )
  n = int(gait["n_samples"])
  frequency = float(gait["frequency"])
  blend_speed = float(gait["blend_speed"])
  gallop_speed = float(gait["gallop_speed"])
  gallop_freq_mult = float(gait["gallop_freq_mult"])
  dir_threshold = float(gait["dir_threshold"])

  t = np.asarray(t, dtype=np.float64)
  vx = np.asarray(vx, dtype=np.float64)
  vy = np.asarray(vy, dtype=np.float64)
  yaw = np.asarray(yaw, dtype=np.float64)

  galloping = np.abs(vx) >= gallop_speed
  freq = frequency * np.where(galloping, gallop_freq_mult, 1.0)

  # frac() keeps the sign of its argument, matching torch.frac; the floor-modulo
  # below is what wraps a negative (time-reversed) phase into the table.
  x = t * freq
  phase = x - np.trunc(x)
  translating = np.abs(vx) >= dir_threshold
  dir_signal = np.where(translating, vx, yaw)
  eff_phase = np.where(dir_signal >= 0.0, phase, -phase)

  pos = eff_phase * n
  i0 = np.floor(pos).astype(np.int64) % n  # floor-modulo: handles negatives.
  i1 = (i0 + 1) % n
  frac = (pos - np.floor(pos))[:, None]

  # Mixed-reference policies lift in place instead of trotting when the command
  # is not translating; the gallop/fast branch overrides either. A direction-
  # split fast gait plays the back table for backward-fast commands (the table
  # ships pre-reversed, so the shared phase reversal above plays it forward).
  if lift is None:
    slow_lo, slow_hi = trot[i0], trot[i1]
  else:
    tr = translating[:, None]
    slow_lo = np.where(tr, trot[i0], lift[i0])
    slow_hi = np.where(tr, trot[i1], lift[i1])
  if gallop_back is None:
    fast_lo, fast_hi = gallop[i0], gallop[i1]
  else:
    fw = (dir_signal >= 0.0)[:, None]
    fast_lo = np.where(fw, gallop[i0], gallop_back[i0])
    fast_hi = np.where(fw, gallop[i1], gallop_back[i1])
  g = galloping[:, None]
  lo = np.where(g, fast_lo, slow_lo)
  hi = np.where(g, fast_hi, slow_hi)
  gait_ref = lo * (1.0 - frac) + hi * frac

  offset = gait_ref - np.asarray(default_joint_pos, dtype=np.float64)
  speed = np.linalg.norm(np.stack([vx, vy, yaw], axis=-1), axis=-1)[:, None]
  blend = np.clip(speed / blend_speed, 0.0, 1.0)
  return blend * offset


def jump_reference_offset_numpy(
  jump: dict[str, Any],
  default_joint_pos: np.ndarray,
  t: np.ndarray,
) -> np.ndarray:
  """NumPy mirror of the on-robot one-shot jump reference.

  ``t`` is seconds since the jump was triggered; idle is ``t = 0`` (the clamp
  pins the phase at ``phase_start``, the mid-stance crouch) and any ``t`` past
  the cycle end holds the landing pose, one full wrap later -- the same crouch.
  The C++ ``compute_jump_reference_offset`` transcribes this; keep the two (and
  training's :func:`~mjlab.tasks.pupper_gait.mdp.jump.jump_reference_offset`)
  in step.
  """
  table = np.asarray(jump["jump_table"], dtype=np.float64)
  n = int(jump["n_samples"])
  frequency = float(jump["frequency"])
  crouch_hold_s = float(jump["crouch_hold_s"])
  phase_start = float(jump["phase_start"])

  t = np.asarray(t, dtype=np.float64)
  phase = phase_start + np.clip((t - crouch_hold_s) * frequency, 0.0, 1.0)

  pos = phase * n
  i0 = np.floor(pos).astype(np.int64) % n
  i1 = (i0 + 1) % n
  frac = (pos - np.floor(pos))[:, None]
  gait_ref = table[i0] * (1.0 - frac) + table[i1] * frac
  return gait_ref - np.asarray(default_joint_pos, dtype=np.float64)


def mixed_jump_reference_offset_numpy(
  gait: dict[str, Any],
  jump_slot: dict[str, Any],
  default_joint_pos: np.ndarray,
  t: np.ndarray,
  vx: np.ndarray,
  vy: np.ndarray,
  yaw: np.ndarray,
  slot_start: np.ndarray,
) -> np.ndarray:
  """NumPy mirror of the on-robot MixedGaitsJump composite reference.

  ``slot_start`` is the active/pending jump-slot start time per sample (same
  clock as ``t``); pass ``+inf`` for no jump. Outside the active window this
  is exactly :func:`reference_offset_numpy`; inside, the jump table plays over
  ``playback_s`` (phase clamped at 1) and a ``cross_fade_s`` linear fade at
  each window edge blends with the mixed reference, zero exactly at the
  boundaries. The C++ transcribes this -- keep the two in step with the
  training-time :func:`~mjlab.tasks.pupper_gait.mdp.mixed_jump.mixed_jump_reference_offset`.
  """
  base = reference_offset_numpy(gait, default_joint_pos, t, vx, vy, yaw)

  table = np.asarray(jump_slot["jump_table"], dtype=np.float64)
  n = int(jump_slot["n_samples"])
  playback_s = float(jump_slot["playback_s"])
  active_s = float(jump_slot["active_s"])
  fade_s = float(jump_slot["cross_fade_s"])

  t = np.asarray(t, dtype=np.float64)
  slot_start = np.asarray(slot_start, dtype=np.float64)
  t_in = t - slot_start
  in_window = (t_in >= 0.0) & (t_in < active_s)

  phase = np.clip(np.where(in_window, t_in, 0.0) / playback_s, 0.0, 1.0)
  pos = phase * n
  i0 = np.floor(pos).astype(np.int64) % n
  i1 = (i0 + 1) % n
  frac = (pos - np.floor(pos))[:, None]
  jump_ref = table[i0] * (1.0 - frac) + table[i1] * frac
  jump_off = jump_ref - np.asarray(default_joint_pos, dtype=np.float64)

  fade = np.clip(
    np.minimum(
      np.where(in_window, t_in, 0.0), active_s - np.where(in_window, t_in, 0.0)
    )
    / fade_s,
    0.0,
    1.0,
  )[:, None]
  slot_ref = fade * jump_off + (1.0 - fade) * base
  return np.where(in_window[:, None], slot_ref, base)


def gait_reference_offset_torch(
  gait: dict[str, Any],
  default_joint_pos: torch.Tensor,
  t: torch.Tensor,
  vx: torch.Tensor,
  vy: torch.Tensor,
  yaw: torch.Tensor,
) -> torch.Tensor:
  """Training-time (torch) reference offset, driven by an exported gait block.

  Used by the parity test to compare the deploy path against the exact function
  the policy was trained with, without needing a live env.
  """
  from mjlab.tasks.pupper_gait.mdp.gait import reference_offset_from_tables

  device = default_joint_pos.device
  trot = torch.tensor(gait["trot_table"], dtype=torch.float32, device=device)
  gallop = torch.tensor(gait["gallop_table"], dtype=torch.float32, device=device)
  return reference_offset_from_tables(
    trot,
    gallop,
    t,
    vx,
    vy,
    yaw,
    default_joint_pos,
    frequency=float(gait["frequency"]),
    blend_speed=float(gait["blend_speed"]),
    gallop_speed=float(gait["gallop_speed"]),
    gallop_freq_mult=float(gait["gallop_freq_mult"]),
  )
