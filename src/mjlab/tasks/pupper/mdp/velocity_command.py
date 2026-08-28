"""Pupper velocity command with an added pure-turn mode.

Extends mjlab's ``UniformVelocityCommand`` (which already provides forward-only
"walk straight" commands via ``rel_forward_envs``) with ``rel_turn_envs``: a
fraction of environments that get a pure yaw command (zero linear velocity, a
non-trivial turn rate), so the policy sees plenty of turn-in-place and
straight-line examples rather than only random x/y/yaw mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class PupperVelocityCommand(UniformVelocityCommand):
  cfg: "PupperVelocityCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: "PupperVelocityCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.is_turn_env = torch.zeros_like(self.is_standing_env)
    # Heading (yaw) captured each time a command is (re)sampled, so a no-yaw
    # command can be rewarded for holding this heading (anti-drift). Captured
    # lazily on the next update, when heading_w is fresh (at resample time -- e.g.
    # during reset -- the root pose may not yet be reflected in the sim data).
    self.heading_hold_target = torch.zeros(self.num_envs, device=self.device)
    self._capture_heading = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _update_command(self) -> None:
    super()._update_command()
    if self._capture_heading.any():
      ids = self._capture_heading.nonzero(as_tuple=False).flatten()
      self.heading_hold_target[ids] = self.robot.data.heading_w[ids]
      self._capture_heading[ids] = False

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._capture_heading[env_ids] = True  # recapture heading on the next update
    r = torch.empty(len(env_ids), device=self.device)
    self.is_turn_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_turn_envs
    turn_ids = env_ids[self.is_turn_env[env_ids]]
    if len(turn_ids) > 0:
      # Pure yaw: zero linear, keep a meaningful turn magnitude (sign preserved).
      yaw = self.vel_command_b[turn_ids, 2]
      sign = torch.where(yaw >= 0, 1.0, -1.0)
      self.vel_command_b[turn_ids, 0] = 0.0
      self.vel_command_b[turn_ids, 1] = 0.0
      self.vel_command_b[turn_ids, 2] = sign * yaw.abs().clamp(min=self.cfg.turn_min)


@dataclass
class PupperVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_turn_envs: float = 0.0
  """Fraction of environments that receive a pure-yaw (turn-in-place) command."""
  turn_min: float = 0.5
  """Minimum absolute yaw rate assigned to turn-only envs [rad/s]."""

  def build(self, env: ManagerBasedRlEnv) -> PupperVelocityCommand:
    return PupperVelocityCommand(self, env)


class PupperMaxSpeedCommand(PupperVelocityCommand):
  """Flat-out forward/backward commands only: ``vx = +/-max_speed``, no y, no yaw.

  For studying what gait emerges when the only objective is speed. The magnitude
  is fixed rather than sampled, so every env is always asking for maximum effort;
  the sign is drawn 50/50 so both directions are learned.

  ``max_speed`` is deliberately set beyond what the robot can reach. It never caps
  anything -- the paired reward is unbounded in achieved speed, not a tracking
  error -- so its only role is to tell the policy, through the command in its
  observation, that the ask is "as fast as you can" rather than a specific target.
  """

  cfg: "PupperMaxSpeedCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    forward = torch.rand(len(env_ids), device=self.device) < 0.5
    self.vel_command_b[env_ids, 0] = torch.where(
      forward, self.cfg.max_speed, -self.cfg.max_speed
    )
    self.vel_command_b[env_ids, 1] = 0.0
    self.vel_command_b[env_ids, 2] = 0.0


@dataclass(kw_only=True)
class PupperMaxSpeedCommandCfg(PupperVelocityCommandCfg):
  """Config for :class:`PupperMaxSpeedCommand`."""

  max_speed: float = 1.5
  """Commanded speed magnitude [m/s]. Beyond the robot's reach on purpose."""

  def build(self, env: ManagerBasedRlEnv) -> PupperMaxSpeedCommand:
    return PupperMaxSpeedCommand(self, env)
