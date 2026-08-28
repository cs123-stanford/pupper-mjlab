"""Pupper-specific reward terms.

These mirror the reward functions in the CS 123 ``pupperv3-mjx`` ``rewards.py``
that have no direct mjlab equivalent. Terms that *do* map cleanly (velocity
tracking, torque/accel/action-rate penalties, flat-orientation, termination,
feet air-time / slip) reuse mjlab's built-in functions in the env config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def heading_deviation(
  env: "ManagerBasedRlEnv",
  command_name: str,
  yaw_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Penalize drifting off the heading captured when the command was issued (L2).

  For a straight-line / lateral / standing command (|yaw_cmd| < ``yaw_threshold``)
  returns the squared heading error -- 0 when on heading, growing quadratically
  (severely) as it diverges. When a turn is commanded it returns 0 (no penalty),
  so the penalty is zero in both acceptable cases (holding heading OR turning-
  when-told) and only bites on unwanted turning. Use with a negative weight. The
  target heading is captured by ``PupperVelocityCommand`` at each resample.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  target = command_term.heading_hold_target  # type: ignore[attr-defined]
  error = wrap_to_pi(asset.data.heading_w - target)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  straight = command[:, 2].abs() < yaw_threshold
  return torch.square(error) * straight.float()


class pupper_variable_posture:
  """Speed-graded posture reward (exp(-mean(error^2/std^2))) with four regimes.

  Like the shared ``variable_posture`` (standing/walking/running by
  ``linear + |yaw|`` speed) but adds a "turning-in-place" regime: when
  translation is small but yaw is large (``linear < turn_linear_max`` and
  ``angular > turn_angular_min``), ``std_turning`` is used instead of the
  walking/running std. This frees the legs a spin actually needs (abduction,
  knee) while still constraining the rest, independent of how the raw combined
  speed would otherwise classify a fast spin as "running".
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    def _resolve(key: str) -> torch.Tensor:
      _, _, values = resolve_matching_names_values(
        data=cfg.params[key], list_of_strings=joint_names
      )
      return torch.tensor(values, device=env.device, dtype=torch.float32)

    self.std_standing = _resolve("std_standing")
    self.std_walking = _resolve("std_walking")
    self.std_running = _resolve("std_running")
    self.std_turning = _resolve("std_turning")

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    std_standing,
    std_walking,
    std_running,
    std_turning,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    turn_linear_max: float = 0.1,
    turn_angular_min: float = 0.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running, std_turning  # Resolved in __init__.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing = total_speed < walking_threshold
    turning = (
      ~standing & (linear_speed < turn_linear_max) & (angular_speed > turn_angular_min)
    )
    walking = ~standing & ~turning & (total_speed < running_threshold)
    running = ~standing & ~turning & (total_speed >= running_threshold)

    std = (
      self.std_standing * standing.float().unsqueeze(1)
      + self.std_turning * turning.float().unsqueeze(1)
      + self.std_walking * walking.float().unsqueeze(1)
      + self.std_running * running.float().unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def lin_vel_z_l2(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Penalize vertical base linear velocity (L2)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Penalize roll/pitch base angular velocity (L2)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def base_height(
  env: "ManagerBasedRlEnv",
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward holding the base at a constant target height, exp(-dz^2/std^2).

  Keeps the torso at its nominal standing height so the dog can't satisfy a yaw
  command by crouching/pivoting in place instead of taking real turning steps.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  return torch.exp(-torch.square(height - target_height) / std**2)


def _command_below(env: "ManagerBasedRlEnv", command_name: str, threshold: float):
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :3], dim=1) < threshold


def stand_still_pose(
  env: "ManagerBasedRlEnv",
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Penalize joint deviation from the default pose at near-zero command (L1)."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  ids = asset_cfg.joint_ids
  dev = torch.sum(
    torch.abs(asset.data.joint_pos[:, ids] - default_joint_pos[:, ids]), dim=1
  )
  return dev * _command_below(env, command_name, command_threshold)


def stand_still_joint_velocity(
  env: "ManagerBasedRlEnv",
  command_name: str,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Penalize joint velocity at near-zero command (L1)."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  vel = torch.sum(torch.abs(asset.data.joint_vel[:, ids]), dim=1)
  return vel * _command_below(env, command_name, command_threshold)


def joint_deviation_l2(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Penalize the selected joints toward zero angle (L2).

  Mirrors the notebook's ``reward_abduction_angle``, which regularizes the
  ``leg_*_2`` joints (the ``[1::3]`` slice of the joint vector) toward zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  return torch.sum(torch.square(asset.data.joint_pos[:, ids]), dim=1)


def track_yaw_velocity(
  env: "ManagerBasedRlEnv",
  sigma: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Yaw-rate tracking, ``exp(-(cmd_yaw - yaw_rate)^2 / sigma)``.

  A direct port of ``pupperv3_mjx.rewards.reward_tracking_ang_vel``, replacing the
  shared velocity-task term for Pupper. Two deliberate differences from that term:

  * **Yaw only.** The shared term adds the squared roll/pitch rate to the yaw
    error, so body wobble -- unavoidable in a trot -- consumes most of the budget:
    measured mid-trot it was 59% of the total error, capping the reward at 0.12
    even with *perfect* yaw tracking, versus 0.35 under this form for merely
    mediocre yaw. That left almost no gradient for the thing the term is named
    after. Roll/pitch is still penalized by ``ang_vel_xy_l2``, exactly as in the
    reference, so dropping it here removes a double penalty rather than a check.
  * **``sigma`` divides the error directly**, not squared -- matching the
    reference's ``exp(-error / tracking_sigma)``. It is deliberately not called
    ``std``, since it is not one.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  yaw_rate = asset.data.root_link_ang_vel_b[:, 2]
  return torch.exp(-torch.square(command[:, 2] - yaw_rate) / sigma)


def forward_speed(
  env: "ManagerBasedRlEnv",
  command_name: str,
  clip: float = 3.0,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward body-frame speed along the commanded direction. No target to sit at.

  Replaces velocity *tracking* for the max-speed study task: a tracking reward
  saturates once the commanded speed is reached, so it can only ever answer "the
  fastest gait that hits X m/s". This keeps paying more the faster the robot goes,
  so the policy converges on whatever the hardware's actual limit is.

  Floored at zero rather than allowed to go negative -- moving the wrong way earns
  nothing instead of being punished, which keeps the term from dominating the
  regularizers early on when the policy is still flailing. ``clip`` bounds it well
  above any plausible speed, so it never binds in practice but the reward cannot
  run away if the sim diverges.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  direction = torch.sign(command[:, 0])
  return torch.clamp(direction * asset.data.root_link_lin_vel_b[:, 0], 0.0, clip)
