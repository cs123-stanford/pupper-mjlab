"""Pupper joint-position action with command latency.

The motor target the policy emits does not reach the actuator instantly: on the
robot it crosses the control bus, which the deploy MuJoCo sim models as
``command_latency_timesteps: 8`` (~15 ms at the 520 Hz control rate) and which the
MJX training pipeline randomizes as ``latency_distribution = [0.2, 0.8]`` -- 20%
no lag, 80% one control step.

Only the *applied* target is delayed. ``raw_action`` is left untouched, so the
``last_action`` observation and the action-rate reward both see the policy's own
undelayed output, matching ``state.info["last_act"] = action`` in the MJX env and
the on-robot controller (verified against hardware recordings: the deployed
``last_action`` equals the previous policy output to 1.4e-6).

mjlab's :class:`~mjlab.actuator.ActuatorCfg` also offers a command delay, but it
is measured in *physics* timesteps and sampled uniformly, so it cannot express a
bimodal per-control-step distribution like ``[0.2, 0.8]``. This term applies the
lag at the env-step granularity the reference pipeline uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.tasks.pupper.mdp.latency import (
  PUPPER_ACTION_LATENCY_DIST,
  LatencyBuffer,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class DelayedJointPositionAction(JointPositionAction):
  """Joint position control whose applied target is stochastically delayed."""

  cfg: "DelayedJointPositionActionCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(
    self, cfg: "DelayedJointPositionActionCfg", env: "ManagerBasedRlEnv"
  ) -> None:
    super().__init__(cfg=cfg, env=env)
    self._buffer = LatencyBuffer(
      cfg.latency_distribution, env.num_envs, (self.action_dim,), env.device
    )
    self._delayed_actions = torch.zeros_like(self._processed_actions)

  @property
  def applied_actions(self) -> torch.Tensor:
    """The processed target actually sent to the actuators this step.

    Equals ``processed_actions`` when the sampled lag is 0 and the previous step's
    target otherwise. Privileged: the robot cannot observe which lag its own bus
    applied, so this must not reach the actor.
    """
    return self._delayed_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    # Runs once per env step, so this is where the lag advances. apply_actions
    # runs once per physics step and must not push the buffer.
    super().process_actions(actions)
    self._delayed_actions = self._buffer.step(self._processed_actions)

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._delayed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._buffer.reset(env_ids)
    self._delayed_actions[env_ids] = 0.0


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position control with a categorical command latency.

  ``latency_distribution[i]`` is the probability of an ``i``-env-step lag,
  resampled every step and per environment. The default matches the CS 123 MJX
  training config.
  """

  latency_distribution: Sequence[float] = field(
    default_factory=lambda: PUPPER_ACTION_LATENCY_DIST
  )

  def build(self, env: "ManagerBasedRlEnv") -> DelayedJointPositionAction:
    return DelayedJointPositionAction(self, env)
