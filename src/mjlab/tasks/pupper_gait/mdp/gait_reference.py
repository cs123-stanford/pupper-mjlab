"""Foot-space trot reference ported from the CS 123 Pupper IK controller.

The real Pupper trot is defined as per-leg triangular *foot* trajectories in
Cartesian space, converted to joint angles by inverse kinematics. This module
ports that controller's forward kinematics, gradient-descent IK, and triangle
interpolation, and precomputes a phase -> 12-joint-angle reference table.

Joint / leg order matches mjlab's ``JOINT_NAMES`` and the ROS controller:
    FR(0-2), FL(3-5), BR(6-8), LB(9-11).
The diagonal trot phasing (RF+LB together, LF+RB together) is baked into each
leg's keyframe ordering, so a single global phase indexes all four legs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  get_spec,
)

# Captured fast gaits: the reach ("fast forward") and reach_back ("fast
# backward") references are no longer IK-generated -- they are the emergent
# gaits a trained MixedGaits policy (run al7sdood, +-1.5 m/s commands,
# velocity-dominant reward) actually performed at +1.5 and -1.5 m/s, recorded
# phase-locked to the reference clock (cycle-to-cycle dispersion < 6 mrad).
# Each left/right leg pair is mirror-averaged (all three joints, one shared
# phase shift) before shipping: the raw capture's worst asymmetry was the
# leg roll -- motor 2, the joint that moves the foot in/out -- with pair sums
# of 0.44-0.68 rad sustained, i.e. one leg tucked in while its partner
# splayed out. Regenerate with the capture script recorded in the npz's
# metadata fields whenever a better fast policy exists.
_CAPTURED_GAITS_FILE = Path(__file__).with_name("captured_reach_gaits.npz")
_CAPTURED_KEYS = {"reach": "reach_fwd", "reach_back": "reach_back"}

# Captured jump: the emergent countermovement of run zuy6c85c (trained on the
# IK pronk + drive shaping below), captured from the loading dip through the
# landing, left/right mirror-averaged (1.58 rad of asymmetry removed) and
# loop-closed onto its own starting crouch. One-shot, not periodic: row 0 is
# the near-stance hold pose, one playback is one jump, and the wrap
# interpolation lands the final settle back on row 0. When this file exists
# the jump task trains against it (at 1/duration_s for its clock, phase
# starting at 0) instead of the shaped IK pronk; delete or rename the file to
# fall back. Regenerate with the capture script recorded in its provenance.
_CAPTURED_JUMP_FILE = Path(__file__).with_name("captured_jump_gait.npz")


def jump_capture_duration_s() -> float | None:
  """Playback duration of the captured jump [s], or None without a capture."""
  if not _CAPTURED_JUMP_FILE.exists():
    return None
  return float(np.load(_CAPTURED_JUMP_FILE)["duration_s"])


def rotation_x(a):
  return np.array(
    [
      [1, 0, 0, 0],
      [0, np.cos(a), -np.sin(a), 0],
      [0, np.sin(a), np.cos(a), 0],
      [0, 0, 0, 1],
    ]
  )


def rotation_y(a):
  return np.array(
    [
      [np.cos(a), 0, np.sin(a), 0],
      [0, 1, 0, 0],
      [-np.sin(a), 0, np.cos(a), 0],
      [0, 0, 0, 1],
    ]
  )


def rotation_z(a):
  return np.array(
    [
      [np.cos(a), -np.sin(a), 0, 0],
      [np.sin(a), np.cos(a), 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]
  )


def translation(x, y, z):
  return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def _fr_leg_fk(theta):
  t01 = translation(0.075, -0.0835, 0) @ rotation_x(1.5708) @ rotation_z(theta[0])
  t12 = rotation_y(-1.5708) @ rotation_z(theta[1])
  t23 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(theta[2])
  t3e = translation(0.06231, -0.06216, 0.018)
  return (t01 @ t12 @ t23 @ t3e)[:3, 3]


def _fl_leg_fk(theta):
  t01 = translation(0.075, 0.0835, 0) @ rotation_x(1.5708) @ rotation_z(-theta[0])
  t12 = rotation_y(-1.5708) @ rotation_z(theta[1])
  t23 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(-theta[2])
  t3e = translation(0.06231, -0.06216, -0.018)
  return (t01 @ t12 @ t23 @ t3e)[:3, 3]


def _br_leg_fk(theta):
  t01 = translation(-0.075, -0.0725, 0) @ rotation_x(1.5708) @ rotation_z(theta[0])
  t12 = rotation_y(-1.5708) @ rotation_z(theta[1])
  t23 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(theta[2])
  t3e = translation(0.06231, -0.06216, 0.018)
  return (t01 @ t12 @ t23 @ t3e)[:3, 3]


def _lb_leg_fk(theta):
  t01 = translation(-0.075, 0.0725, 0) @ rotation_x(1.5708) @ rotation_z(-theta[0])
  t12 = rotation_y(-1.5708) @ rotation_z(theta[1])
  t23 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(-theta[2])
  t3e = translation(0.06231, -0.06216, -0.018)
  return (t01 @ t12 @ t23 @ t3e)[:3, 3]


_FK = (_fr_leg_fk, _fl_leg_fk, _br_leg_fk, _lb_leg_fk)


# Per-leg body-frame hip offsets, FR/FL/BR/LB order.
_HIP_OFFSET = np.array(
  [[0.06, -0.09, 0], [0.06, 0.09, 0], [-0.11, -0.09, 0], [-0.11, 0.09, 0]]
)

# Per-leg touchdown phase in [0, 1) -- where in the cycle each leg lands (is at
# the front of its stance) -- FR/FL/BR/LB order. This sets the footfall pattern.
#   trot   : diagonal pairs (FR+LB, FL+BR) a half cycle apart.
#   gallop : rotatory footfall order, left lead -- the back pair lands first (BL
#            then BR), then the front pair (FR then FL): BL -> BR -> FR -> FL.
#
#            NOTE: this is the footfall *ordering* only, not a true gallop. The
#            duty factor is fixed at 4/6 by ``_base_cycle`` (five stance keyframes,
#            one swing), so at least two feet are always down and there is no
#            suspension phase anywhere in the cycle -- the same 2.67 mean feet-down
#            as the trot. A real gallop needs duty < 0.5, which means restructuring
#            the base cycle, not just re-phasing it. The gait is currently disabled
#            (see ``GAIT_GALLOP_SPEED``); it is not trackable by Pupper's 3 Nm
#            actuators at the 0.08 m stride and 2x cadence it was configured with.
#   cheetah: the rotary gallop as the animal actually runs it -- non-lead fore,
#            lead fore, non-lead hind, lead hind. Note the "gallop" entry above has
#            this inverted (hind pair first); correcting the order alone drops peak
#            actuator demand 21%. Paired with a long swing (see _GAIT_SWING_KF) it
#            is the only configuration here with a real suspension phase.
_GAIT_TOUCHDOWN: dict[str, tuple[float, float, float, float] | None] = {
  "trot": (0.0, 0.5, 0.5, 0.0),
}

# Fore-aft half-stride [m] per gait. A scalar is symmetric; a (front, back)
# pair would reach further forward than back.
_GAIT_STRIDE: dict[str, float | tuple[float, float] | None] = {
  "trot": 0.05,
}

# Swing keyframes per gait (default 1). One keyframe crams the whole leg return
# into a third of the cycle (duty factor 4/6), which is both unlike a galloping
# animal and what saturates the actuator. Nine spreads it to duty 0.29 and buys a
# suspension phase over 23% of the cycle.
_GAIT_SWING_KF: dict[str, int] = {
  "trot": 1,
}

# Per-leg fore-aft stance bias [m] (FR, FL, BR, LB); empty = none.
_GAIT_X_SHIFT: dict[str, tuple[float, float, float, float]] = {}

# Per-leg lateral stance spread [m] (FR, FL, BR, LB); empty = none.
_GAIT_Y_SHIFT: dict[str, tuple[float, float, float, float]] = {}

_STANCE_Z = -0.14
_SWING_Z = -0.05

# Stance depth per gait: how far below the hip the foot is planted. This, not
# the joint limits, is what bounds forward reach: the leg is 0.1734 m from the
# hip, so a foot planted 0.14 m down has only 0.102 m of forward travel left.
# Crouching converts vertical budget into horizontal reach -- worth knowing
# when you design a faster reference.
_GAIT_STANCE_Z: dict[str, float | None] = {
  "trot": -0.14,
}

# Foot lift during swing, measured *from the stance plane* rather than as an
# absolute height. This has to be relative: _SWING_Z is a fixed -0.05, so as the
# stance gets shallower the gap between them closes and the clearance vanishes --
# at a 0.10 m stance the foot would rise 5 cm rather than the 9 the original pair
# gave.
_GAIT_SWING_LIFT: dict[str, float | None] = {
  "trot": 0.09,
}

# Phase-clock multiplier per gait, relative to the base cadence; empty = 1x
# for everything. A faster reference usually wants a quicker cycle.
_GAIT_FREQ_MULT: dict[str, float] = {}


# =============================================================================
# TODO(student): design the lift gait.
#
# StableGait picks its reference from the command: the trot above plays for
# forward and backward walking, and a second gait -- "lift" -- plays whenever
# the command is a pure turn or sidestep. Its job is to keep the feet stepping
# (so the policy can rotate or shift the body between touchdowns) without the
# reference itself dragging the robot anywhere.
#
# The trot entries above are a worked example, and you have built it before:
# ``_interpolate_triangle`` is the keyframe interpolation from lab 3,
# ``_ik_single_leg`` is the same gradient-descent IK, ``_base_cycle`` is your
# Raibert triangle (five stance keyframes sliding back, one lifted swing), and
# the trot's touchdown phases are the leg coordination you chose there.
#
# Before you fill these in, decide -- and write down in your lab doc:
#   * Stride: the trot slides each planted foot backward so the body moves
#     forward. What must the stride be if the reference is not allowed to
#     oppose a turn-in-place?
#   * Touchdown phases: which legs should land together? Does the pairing you
#     picked in lab 3 for forward trotting still make sense when the robot
#     is not going anywhere?
#   * Stance depth and swing lift: the trot's values hold the body at standing
#     height and clear the terrain. Any reason to change them here?
#
# Replace each None with a real value. StableGait refuses to build until you
# do, and the lift tests in tests/test_pupper_gait_export.py un-skip to check
# that your gait deploys. Look at it before you train with it:
#
#   uv run python -m mjlab.tasks.pupper_gait.visualize_reference --gait lift
# =============================================================================
_GAIT_TOUCHDOWN["lift"] = None  # (FR, FL, BR, LB) touchdown phases in [0, 1)
_GAIT_STRIDE["lift"] = None  # fore-aft half-stride [m]
_GAIT_STANCE_Z["lift"] = None  # foot depth below the hip during stance [m]
_GAIT_SWING_LIFT["lift"] = None  # foot rise above the stance plane in swing [m]


def _base_cycle(
  stride: float | tuple[float, float],
  n_swing: int = 1,
  stance_z: float | None = None,
  swing_lift: float | None = None,
) -> np.ndarray:
  """Foot cycle: stance slides front->back over 5 keyframes, then a lifted swing.

  ``n_swing`` keyframes carry the foot back to the front of stance. One gives the
  original 6-keyframe cycle (duty factor 4/6); more lengthens the swing relative to
  stance, lowering the duty factor toward the sub-0.5 range a galloping animal uses
  and spreading the leg's return over more control steps.
  """
  front, back = stride if isinstance(stride, tuple) else (stride, stride)
  z = _STANCE_Z if stance_z is None else stance_z
  rise = (_SWING_Z - _STANCE_Z) if swing_lift is None else swing_lift
  stance = [
    [front, 0.0, z],  # touch down (front of stance)
    [0.5 * front, 0.0, z],  # stand 1
    [0.0, 0.0, z],  # stand 2
    [-0.5 * back, 0.0, z],  # stand 3
    [-back, 0.0, z],  # liftoff (back of stance)
  ]
  if n_swing == 1:
    return np.array(stance + [[0.0, 0.0, z + rise]])
  lift = rise
  swing = [
    [
      -back + (front + back) * (k / (n_swing + 1)),
      0.0,
      z + lift * np.sin(np.pi * k / (n_swing + 1)),
    ]
    for k in range(1, n_swing + 1)
  ]
  return np.array(stance + swing)


def _interpolate_triangle(t: float, positions: np.ndarray) -> np.ndarray:
  """Linearly interpolate the looping keyframe trajectory at phase t in [0, 1)."""
  n = len(positions)
  t = t * n
  progress = t - int(t)
  start = int(t) % n
  end = (start + 1) % n
  return positions[start] + progress * (positions[end] - positions[start])


def _ik_single_leg(target_ee, leg_index, initial_guess):
  fk = _FK[leg_index]

  def cost(theta):
    return float(np.sum((fk(theta) - target_ee) ** 2))

  theta = np.array(initial_guess, dtype=np.float64)
  lr, max_iter, tol, eps = 10.0, 100, 1e-4, 1e-3
  for _ in range(max_iter):
    grad = np.zeros(3)
    for i in range(3):
      tp, tm = theta.copy(), theta.copy()
      tp[i] += eps
      tm[i] -= eps
      grad[i] = (cost(tp) - cost(tm)) / (2 * eps)
    theta -= lr * grad
    if np.abs(fk(theta) - target_ee).mean() < tol:
      break
  return theta


_TABLE_CACHE: dict[tuple[int, str], np.ndarray] = {}


def _clamp_to_joint_limits(table: np.ndarray) -> np.ndarray:
  model = get_spec().compile()
  limits = np.array([model.joint(n).range for n in JOINT_NAMES])
  return np.clip(table, limits[:, 0], limits[:, 1])


def build_joint_reference_table(n_samples: int = 100, gait: str = "trot") -> np.ndarray:
  """Precompute the phase -> (n_samples, 12) joint-angle reference for a gait.

  Each leg reads the shared foot cycle at ``(phase - touchdown[leg]) mod 1`` (so
  it lands at its touchdown phase), then the per-leg hip offset (and, for the
  gallop, the fore-aft spread) is added before IK.
  """
  key = (n_samples, gait)
  if key in _TABLE_CACHE:
    return _TABLE_CACHE[key]
  if gait == "jump" and _CAPTURED_JUMP_FILE.exists():
    src = np.load(_CAPTURED_JUMP_FILE)["jump"]
    if len(src) != n_samples:
      # Non-periodic resample: the jump is one-shot, so the endpoints are the
      # hold pose and the landed settle -- both preserved, no wrap.
      xp = np.arange(len(src)) / (len(src) - 1)
      grid = np.arange(n_samples) / (n_samples - 1)
      src = np.stack([np.interp(grid, xp, src[:, j]) for j in range(12)], axis=1)
    table = _clamp_to_joint_limits(np.ascontiguousarray(src))
    _TABLE_CACHE[key] = table
    return table
  if gait in _CAPTURED_KEYS and _CAPTURED_GAITS_FILE.exists():
    src = np.load(_CAPTURED_GAITS_FILE)[_CAPTURED_KEYS[gait]]
    if len(src) != n_samples:
      # Periodic linear resample onto the requested grid.
      xp = np.arange(len(src) + 1) / len(src)
      fp = np.vstack([src, src[:1]])
      grid = np.arange(n_samples) / n_samples
      src = np.stack([np.interp(grid, xp, fp[:, j]) for j in range(12)], axis=1)
    table = np.ascontiguousarray(src)
    if gait == "reach_back":
      # The lookup pipeline time-reverses the phase for backward commands
      # (eff_phase = -phase, floor-mod wrapped). The capture is indexed by the
      # true forward-running phase, so store it pre-reversed -- the pipeline's
      # reversal then plays the recording forward exactly as captured.
      table = np.roll(table[::-1], 1, axis=0)
    _TABLE_CACHE[key] = table
    return table
  if gait not in _GAIT_TOUCHDOWN:
    # Only the trot generator ships. Every other slot -- the fast gait,
    # anything a task schedules -- plays the trot until you provide something
    # better: drop a captured_*.npz next to this file (see the capture notes
    # above) or extend the generator tables. Finding a reference that beats
    # the trot here is the optional lab.
    table = build_joint_reference_table(n_samples, "trot")
    _TABLE_CACHE[key] = table
    return table
  stride = _GAIT_STRIDE.get(gait)
  touchdown = _GAIT_TOUCHDOWN[gait]
  stance_z = _GAIT_STANCE_Z.get(gait)
  swing_lift = _GAIT_SWING_LIFT.get(gait)
  if touchdown is None or stride is None or stance_z is None or swing_lift is None:
    raise NotImplementedError(
      f"The '{gait}' gait is declared but not designed yet: fill in the "
      f"TODO(student) block in {Path(__file__).name} (touchdown phases, "
      "stride, stance depth, swing lift). Then look at it before training "
      "with it:  uv run python -m mjlab.tasks.pupper_gait.visualize_reference "
      f"--gait {gait}"
    )
  cycle = _base_cycle(
    stride,
    _GAIT_SWING_KF.get(gait, 1),
    stance_z,
    swing_lift,
  )
  x_shift = _GAIT_X_SHIFT.get(gait, (0.0, 0.0, 0.0, 0.0))
  y_shift = _GAIT_Y_SHIFT.get(gait, (0.0, 0.0, 0.0, 0.0))
  table = np.zeros((n_samples, 12))
  guesses = [np.zeros(3) for _ in range(4)]
  for i in range(n_samples):
    t = i / n_samples
    for leg in range(4):
      foot = _interpolate_triangle((t - touchdown[leg]) % 1.0, cycle)
      target = foot + _HIP_OFFSET[leg]
      target[0] += x_shift[leg]
      target[1] += y_shift[leg]
      theta = _ik_single_leg(target, leg, guesses[leg])
      table[i, 3 * leg : 3 * leg + 3] = theta
      guesses[leg] = theta  # warm-start next phase for a continuous branch
  _TABLE_CACHE[key] = table
  return table
