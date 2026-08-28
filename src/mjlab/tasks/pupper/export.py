"""Export a trained mjlab Pupper policy to the on-robot ``neural_controller`` JSON.

The robot loads the policy with ``RTNeural::json_parser::parseJson`` and feeds it a
720-dim observation that is **frame-major, newest-first** (see
``neural_controller.cpp``: the newest 36-dim frame occupies indices ``[0:36]``).
mjlab, by contrast, produces the history **oldest-first**, and normalizes
observations with a running mean/std. This exporter reconciles both:

1. Fold the empirical observation normalization into the first linear layer.
2. Reverse the 20 history frame-blocks of the first layer's input columns, so the
   exported network consumes the robot's newest-first ordering.
3. Emit the RTNeural JSON schema (kernels transposed to ``[in][out]``, ELU hidden
   layers, an empty final activation = linear, matching rsl_rl's un-squashed
   Gaussian mean) plus the deploy metadata the controller reads.

Use :func:`json_forward` (a NumPy re-implementation of the RTNeural dense stack) to
check parity against the source policy.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn as nn

# Deploy contract version stamped into every exported JSON. Version 2 adds the
# declarative ``obs_spec`` (the robot assembles the frame from named components
# instead of a compiled-in layout), the optional ``command_clip`` block, and the
# direction-split fast gait table (``fast_back_table``). The robot-side
# controller keeps reading version-1 JSONs (no ``contract_version`` key) via its
# legacy implicit 36/48-dim layouts.
CONTRACT_VERSION = 2

# The base (velocity-task) frame layout, in order. These names are the fixed
# vocabulary the robot-side observation assembler understands; adding a new
# component means adding a provider to ``neural_controller`` once, after which
# any layout built from the menu deploys without robot-side changes.
BASE_OBS_COMPONENTS: tuple[tuple[str, int], ...] = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("command", 3),
  ("desired_world_z", 3),
  ("joint_pos_rel", 12),
  ("last_action", 12),
)


def _linear_layers(mlp: nn.Module) -> list[nn.Linear]:
  layers = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
  if not layers:
    raise ValueError("No nn.Linear layers found in the actor MLP.")
  return layers


def _normalizer_mean_std(
  normalizer: nn.Module, in_dim: int
) -> tuple[np.ndarray, np.ndarray]:
  """Return (mean, std_eff) where the normalization is (x - mean) / std_eff.

  For rsl_rl's ``EmpiricalNormalization`` the effective denominator is
  ``std + eps`` (eps defaults to 1e-2), which must be folded exactly. An
  ``Identity`` normalizer maps to mean 0 / std 1.
  """
  mean_buf = getattr(normalizer, "_mean", None)
  std_buf = getattr(normalizer, "_std", None)
  if mean_buf is None or std_buf is None:
    return np.zeros(in_dim), np.ones(in_dim)
  eps = float(getattr(normalizer, "eps", 0.0))
  mean = mean_buf.detach().cpu().numpy().reshape(-1)
  std = std_buf.detach().cpu().numpy().reshape(-1)
  return mean, std + eps


def _frame_reversal_index(history_length: int, single_obs_dim: int) -> np.ndarray:
  """Column permutation mapping robot (newest-first) order to mjlab (oldest-first).

  Exported column ``g*D + d`` (robot frame g, 0 = newest) reads folded-layer
  column ``(H-1-g)*D + d`` (mjlab frame H-1-g).
  """
  h, d = history_length, single_obs_dim
  return np.concatenate(
    [np.arange((h - 1 - g) * d, (h - 1 - g) * d + d) for g in range(h)]
  )


def convert_pupper_policy(
  actor: nn.Module,
  *,
  action_scale: float,
  kp: float,
  kd: float,
  default_joint_pos: Sequence[float],
  joint_lower_limits: Sequence[float],
  joint_upper_limits: Sequence[float],
  observation_history: int,
  single_obs_dim: int,
  use_imu: bool = True,
  obs_components: Sequence[tuple[str, int]] = BASE_OBS_COMPONENTS,
  command_clip: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
  """Convert a trained actor (``MLPModel``-like: has ``.mlp`` and ``.obs_normalizer``)
  into the deploy JSON dict.

  ``obs_components`` is the ordered frame layout, emitted as the ``obs_spec``
  block; component sizes must sum to ``single_obs_dim``. ``command_clip`` maps
  ``vx``/``vy``/``wz`` to (lo, hi) bounds the robot clamps the teleop command
  to before it reaches the observation or the gait reference; None omits the
  block (no robot-side clamp beyond the teleop scaling).
  """
  components = [(str(n), int(s)) for n, s in obs_components]
  comp_total = sum(s for _, s in components)
  if comp_total != single_obs_dim:
    raise ValueError(
      f"obs_spec components sum to {comp_total}, but single_obs_dim is "
      f"{single_obs_dim}: {components}"
    )
  if command_clip is not None:
    unknown = set(command_clip) - {"vx", "vy", "wz"}
    if unknown:
      raise ValueError(f"command_clip has unknown axes: {sorted(unknown)}")
    for axis, (lo, hi) in command_clip.items():
      if not lo <= hi:
        raise ValueError(f"command_clip.{axis} is inverted: ({lo}, {hi})")

  mlp = cast(nn.Module, actor.mlp)
  normalizer = cast(nn.Module, actor.obs_normalizer)
  linears = _linear_layers(mlp)

  in_dim = linears[0].in_features
  expected = observation_history * single_obs_dim
  if in_dim != expected:
    raise ValueError(
      f"Actor input dim {in_dim} != observation_history*single_obs_dim "
      f"({observation_history}*{single_obs_dim}={expected})."
    )

  mean, std_eff = _normalizer_mean_std(normalizer, in_dim)

  # Fold normalization into layer 0: W0' = W0/std_eff, b0' = b0 - W0 @ (mean/std_eff).
  w0 = linears[0].weight.detach().cpu().numpy()  # (out, in)
  b0 = linears[0].bias.detach().cpu().numpy()  # (out,)
  w0_folded = w0 / std_eff[None, :]
  b0_folded = b0 - w0 @ (mean / std_eff)

  # Reverse history frame-blocks so the exported net consumes newest-first input.
  idx = _frame_reversal_index(observation_history, single_obs_dim)
  w0_folded = w0_folded[:, idx]

  layers = []
  for i, lin in enumerate(linears):
    if i == 0:
      w, b = w0_folded, b0_folded
    else:
      w = lin.weight.detach().cpu().numpy()
      b = lin.bias.detach().cpu().numpy()
    is_final = i == len(linears) - 1
    layers.append(
      {
        "type": "dense",
        "activation": "" if is_final else "elu",  # "" = linear (RTNeural).
        "shape": [None, int(w.shape[0])],
        "weights": [w.T.tolist(), b.tolist()],  # RTNeural wants [in][out].
      }
    )

  out: dict[str, Any] = {
    "contract_version": CONTRACT_VERSION,
    "use_imu": use_imu,
    "control_orientation": True,
    "observation_history": observation_history,
    # Kept alongside obs_spec for older controllers, which size the frame from
    # this (and fall back to their built-in 36 when it too is absent).
    "single_observation_size": int(single_obs_dim),
    # The declarative frame layout: the robot assembles each frame from these
    # named components, in order, newest frame first in the flattened history.
    "obs_spec": {
      "frame_order": "newest_first",
      "history": observation_history,
      "components": [{"name": n, "size": s} for n, s in components],
    },
    "action_scale": float(action_scale),
    "kp": float(kp),
    "kd": float(kd),
    "default_joint_pos": [float(x) for x in default_joint_pos],
    "joint_upper_limits": [float(x) for x in joint_upper_limits],
    "joint_lower_limits": [float(x) for x in joint_lower_limits],
    "in_shape": [None, int(in_dim)],
    "layers": layers,
  }
  if command_clip is not None:
    out["command_clip"] = {
      axis: [float(lo), float(hi)] for axis, (lo, hi) in command_clip.items()
    }
  return out


def command_clip_from_env(env) -> dict[str, tuple[float, float]] | None:
  """The trained command ranges, as the deploy ``command_clip`` block.

  Clamping the teleop command to the ranges the policy actually trained on is
  the default: outside them the policy is extrapolating. Export callers can
  override with a tighter clip (e.g. capping a mixed-gaits policy to its trot
  band for the student default), but a wider one just re-introduces the
  out-of-distribution commands the clip exists to prevent.
  """
  twist = env.cfg.commands.get("twist")
  if twist is None:
    return None
  ranges = twist.ranges
  return {
    "vx": (float(ranges.lin_vel_x[0]), float(ranges.lin_vel_x[1])),
    "vy": (float(ranges.lin_vel_y[0]), float(ranges.lin_vel_y[1])),
    "wz": (float(ranges.ang_vel_z[0]), float(ranges.ang_vel_z[1])),
  }


def export_pupper_policy_from_env(
  actor: nn.Module,
  env,
  *,
  use_imu: bool = True,
  single_obs_dim: int | None = None,
  obs_components: Sequence[tuple[str, int]] | None = None,
  command_clip: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
  """Convenience wrapper: assemble deploy metadata from a live env and export.

  Pulls kp/kd from the compiled model, the default pose and joint limits from the
  robot (in actuated-joint order), and the action scale + history length from the
  Pupper constants / observation manager, so the JSON is always consistent with
  what was trained.

  ``single_obs_dim`` defaults to the velocity task's 36-dim frame; tasks that
  extend the frame (e.g. the gait task, which appends a 12-dim reference offset)
  must pass their own frame size (and matching ``obs_components``) so the history
  length is inferred correctly. ``command_clip`` defaults to the env's trained
  command ranges (see :func:`command_clip_from_env`); pass an explicit mapping to
  override, e.g. to cap a policy below its trained speed band.
  """
  from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
    PUPPER_ACTION_SCALE,
    SINGLE_OBS_DIM,
  )

  if single_obs_dim is None:
    single_obs_dim = SINGLE_OBS_DIM
  if obs_components is None:
    obs_components = BASE_OBS_COMPONENTS
  if command_clip is None:
    command_clip = command_clip_from_env(env)

  robot = env.scene["robot"]
  mj = env.sim.mj_model

  # Joint limits and actuator gains come from the entity's own indexing rather
  # than name lookups: the compiled scene prefixes every element with the entity
  # name ("robot/leg_front_r_1"), so looking up the bare joint name silently
  # returns id -1.
  limits = robot.data.joint_pos_limits[0].detach().cpu().numpy()
  lower = limits[:, 0].tolist()
  upper = limits[:, 1].tolist()

  aids = robot.indexing.ctrl_ids.detach().cpu().numpy()
  kps = mj.actuator_gainprm[aids, 0]
  kds = -mj.actuator_biasprm[aids, 2]
  if not (np.allclose(kps, kps[0]) and np.allclose(kds, kds[0])):
    raise ValueError(
      "The deploy JSON carries a single scalar kp/kd, but the model has "
      f"per-joint gains (kp={kps.tolist()}, kd={kds.tolist()})."
    )

  default_np = robot.data.default_joint_pos[0].detach().cpu().numpy().tolist()

  in_dim = _linear_layers(cast(nn.Module, actor.mlp))[0].in_features
  history = in_dim // single_obs_dim

  return convert_pupper_policy(
    actor,
    action_scale=PUPPER_ACTION_SCALE,
    kp=float(kps[0]),
    kd=float(kds[0]),
    default_joint_pos=default_np,
    joint_lower_limits=lower,
    joint_upper_limits=upper,
    observation_history=history,
    single_obs_dim=single_obs_dim,
    use_imu=use_imu,
    obs_components=obs_components,
    command_clip=command_clip,
  )


def json_forward(policy: dict[str, Any], x: np.ndarray) -> np.ndarray:
  """NumPy re-implementation of RTNeural's dense stack (for parity testing).

  ``x`` is a (batch, in_dim) array in the robot's newest-first order.
  """
  a = np.asarray(x, dtype=np.float64)
  for layer in policy["layers"]:
    w = np.asarray(layer["weights"][0], dtype=np.float64)  # [in][out]
    b = np.asarray(layer["weights"][1], dtype=np.float64)  # [out]
    a = a @ w + b
    act = layer["activation"]
    if act == "elu":
      a = np.where(a > 0.0, a, np.exp(a) - 1.0)
    elif act == "tanh":
      a = np.tanh(a)
    elif act in ("", "linear"):
      pass
    else:
      raise ValueError(f"Unsupported activation in parity forward: {act!r}")
  return a


def to_robot_order(
  x_mjlab: torch.Tensor | np.ndarray, history_length: int, single_obs_dim: int
):
  """Reorder an mjlab (oldest-first) observation into robot (newest-first) order."""
  arr = (
    x_mjlab.detach().cpu().numpy()
    if isinstance(x_mjlab, torch.Tensor)
    else np.asarray(x_mjlab)
  )
  idx = _frame_reversal_index(history_length, single_obs_dim)
  return arr[:, idx]
