"""Pupper-specific observation terms.

The actor observation reproduces the 36-dim layout of the CS 123 ``pupperv3-mjx``
environment exactly, so a policy trained here can be deployed onto the on-robot
``neural_controller`` without changing its observation-construction code:

    [ body_ang_vel(3), projected_gravity(3), command(3),
      desired_world_z(3), joint_pos - default(12), last_action(12) ]

It is emitted as a *single* observation term so that, with history enabled, the
flattened vector is frame-major (each 36-dim frame contiguous) exactly like the
MJX env and the robot. Per-subgroup noise is baked in here (rather than via the
term's ``noise`` field) because the subgroups use different noise scales.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper.mdp.actions import DelayedJointPositionAction
from mjlab.tasks.pupper.mdp.latency import LatencyBuffer

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")

# The IMU latency buffer is owned by the env rather than the observation term:
# the same buffer must be shared by the velocity and gait actor observations, and
# it has to survive across the manager's per-term bookkeeping.
_IMU_LATENCY_ATTR = "_pupper_imu_latency_buffer"

# ang_vel(3) + projected_gravity(3): the dims the robot's IMU actually produces.
_IMU_DIM = 6


def get_imu_latency_buffer(
  env: "ManagerBasedRlEnv", distribution: Sequence[float]
) -> LatencyBuffer:
  """Get (creating on first use) the env's shared IMU latency buffer."""
  buffer = getattr(env, _IMU_LATENCY_ATTR, None)
  if buffer is None or buffer.max_lag != len(distribution) - 1:
    buffer = LatencyBuffer(distribution, env.num_envs, (_IMU_DIM,), env.device)
    setattr(env, _IMU_LATENCY_ATTR, buffer)
  return buffer


def reset_imu_latency(
  env: "ManagerBasedRlEnv", env_ids: torch.Tensor | slice | None
) -> None:
  """Event term: clear IMU latency history for resetting environments.

  A no-op until the first observation is computed, since the buffer is created
  lazily.
  """
  buffer = getattr(env, _IMU_LATENCY_ATTR, None)
  if buffer is not None:
    buffer.reset(env_ids)


@dataclass(frozen=True)
class PupperObsNoise:
  """Per-subgroup uniform noise half-widths, matching the notebook defaults."""

  ang_vel: float = 0.1
  gravity: float = 0.05
  joint_pos: float = 0.05
  last_action: float = 0.01


_DEFAULT_NOISE = PupperObsNoise()


def _uniform(shape, scale: float, device) -> torch.Tensor:
  if scale == 0.0:
    return torch.zeros(shape, device=device)
  return torch.empty(shape, device=device).uniform_(-scale, scale)


def pupper_actor_obs(
  env: "ManagerBasedRlEnv",
  command_name: str = "twist",
  add_noise: bool = True,
  noise: PupperObsNoise = _DEFAULT_NOISE,
  asset_cfg: SceneEntityCfg = _ROBOT,
  imu_latency_distribution: Sequence[float] | None = None,
) -> torch.Tensor:
  """Assemble the deployable 36-dim Pupper observation frame.

  Returns shape (num_envs, 36). History/flattening is handled by the observation
  manager at the group level.

  ``imu_latency_distribution`` models the robot's IMU pipeline latency (see
  :mod:`.latency`). It lags only the 6 IMU dims -- angular velocity and projected
  gravity -- and is applied *after* noise, matching the MJX env. The joint
  encoders are read straight through on the robot, so they are not lagged, and
  ``last_action`` is the policy's own output so it is never lagged either.

  The delay must live here rather than on the observation term, because the
  manager flattens history per term and then concatenates: splitting the IMU into
  its own delayed term would emit a term-major vector and break the deploy frame
  layout, which requires ``[frame0(48), frame1(48), ...]``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  device = asset.data.root_link_ang_vel_b.device
  n = env.num_envs

  ang_vel = asset.data.root_link_ang_vel_b
  gravity = asset.data.projected_gravity_b

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  command = command[:, :3]

  # Desired world-z in body frame. The notebook trains with zero pitch/roll
  # command, so this is the constant upright target; kept in the vector to
  # preserve the deploy layout (control_orientation=True).
  desired_world_z = torch.zeros((n, 3), device=device)
  desired_world_z[:, 2] = 1.0

  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_pos_rel = asset.data.joint_pos - default_joint_pos

  last_action = env.action_manager.action

  if add_noise:
    ang_vel = ang_vel + _uniform(ang_vel.shape, noise.ang_vel, device)
    # Renormalize after perturbing: the robot derives projected gravity from the
    # IMU quaternion, so it is always a unit direction. Leaving the noised vector
    # unnormalized would train the policy on a ~5% norm variation the hardware
    # never produces. Matches the MJX env.
    gravity = gravity + _uniform(gravity.shape, noise.gravity, device)
    gravity = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    joint_pos_rel = joint_pos_rel + _uniform(
      joint_pos_rel.shape, noise.joint_pos, device
    )
    last_action = last_action + _uniform(last_action.shape, noise.last_action, device)

  # IMU latency, post-noise, on the 6 IMU dims only (mirrors the MJX env).
  if imu_latency_distribution is not None and len(imu_latency_distribution) > 1:
    imu = get_imu_latency_buffer(env, imu_latency_distribution).step(
      torch.cat([ang_vel, gravity], dim=-1)
    )
    ang_vel, gravity = imu[:, :3], imu[:, 3:]

  obs = torch.cat(
    [ang_vel, gravity, command, desired_world_z, joint_pos_rel, last_action], dim=-1
  )
  return torch.clip(obs, -100.0, 100.0)


def applied_action(
  env: "ManagerBasedRlEnv", action_name: str = "joint_pos"
) -> torch.Tensor:
  """The joint-position target actually applied this step, after command latency.

  **Privileged -- critic only.** The robot cannot observe which lag its own control
  bus applied, so feeding this to the actor would train on information hardware
  cannot reproduce (and would break the fixed deploy frame layout).

  The value of this to the critic: the command lag is a per-step, per-env coin flip
  (:mod:`.latency`), and ``mdp.last_action`` deliberately reports the policy's raw
  output rather than what was applied (see :mod:`.actions`). With only the raw
  action the value function cannot see the plant's actual input -- the two differ
  on ~80% of steps under ``PUPPER_ACTION_LATENCY_DIST`` -- and that unexplained
  transition noise inflates advantage variance. Supplying both lets the critic
  recover the realized lag itself, since a nonzero ``applied - raw`` means the
  target was stale.
  """
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, DelayedJointPositionAction):
    raise TypeError(
      f"Action term '{action_name}' is a {type(term).__name__}, which applies no "
      "command latency, so 'applied_action' would silently duplicate "
      "'last_action'. Use mdp.last_action, or configure the term as a "
      "DelayedJointPositionActionCfg."
    )
  return term.applied_actions
