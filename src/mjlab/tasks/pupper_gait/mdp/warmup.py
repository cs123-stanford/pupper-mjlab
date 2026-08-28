"""Airborne pretraining phase: learn the reference motion before learning to walk.

Tracking the gait reference and staying upright are two different problems, and
solving them simultaneously from scratch is what makes the early curve slow -- the
policy spends its first thousands of iterations discovering balance while the gait
term is still near its random-action value.

This splits them. For the first ``until_step`` env steps the base is pinned
upright and stationary in the air and every reward except reference tracking is
zeroed, so the only gradient is "match the reference joint trajectory". The legs
swing freely, contacts never fire, and balance is not yet a question. After that
the pin releases, all weights come back, and the policy starts phase two already
knowing the motion.

The pin is a ``"step"``-mode event rather than a gravity change: gravity lives in
the compiled model, and toggling it mid-run means recomputing derived constants on
the fly. Writing the root state every step costs one write and touches nothing the
solver caches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.entity import Entity
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def hold_base_airborne(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  height: float,
  until_step: int,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Step event: pin the base upright and motionless at ``height`` until
  ``until_step`` env steps have elapsed, then become a no-op.

  Runs on every env unconditionally (``"step"`` mode passes ``env_ids=None``).
  """
  del env_ids  # "step" mode always applies to all envs.
  if env.common_step_counter >= until_step:
    return

  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None

  root_states = default_root_state.clone()
  # Keep each env over its own origin so the debug visualizers stay sensible.
  root_states[:, 0:2] += env.scene.env_origins[:, 0:2]
  root_states[:, 2] = height
  # Identity quaternion (w, x, y, z): upright, so bad_orientation cannot fire.
  root_states[:, 3] = 1.0
  root_states[:, 4:7] = 0.0
  root_states[:, 7:] = 0.0  # zero linear and angular velocity
  asset.write_root_state_to_sim(root_states)


class pretrain_rewards:
  """Curriculum: zero every reward except ``keep`` until ``until_step``.

  Must be the **last** curriculum term registered. Others (the drift ramp, the
  command schedule) write their own weights on every compute, so a term that runs
  after this one would undo the zeroing while the warmup is still active.

  On the step the warmup ends, the captured construction-time weights are written
  back once and this stops touching them; whatever other curricula govern those
  terms take over from the next compute.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: "ManagerBasedRlEnv") -> None:
    keep: Sequence[str] = cfg.params["keep"]
    self._until = int(cfg.params["until_step"])
    self._reward_manager = env.reward_manager
    missing = set(keep) - set(self._reward_manager.active_terms)
    if missing:
      raise KeyError(
        f"pretrain_rewards asked to keep unknown reward term(s) {sorted(missing)}; "
        "the warmup would zero everything and train on nothing."
      )
    self._keep = set(keep)
    self._original = {
      name: self._reward_manager.get_term_cfg(name).weight
      for name in self._reward_manager.active_terms
    }
    self._was_active = False

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    keep: Sequence[str],
    until_step: int,
  ) -> dict[str, torch.Tensor]:
    del env_ids, keep, until_step
    active = env.common_step_counter < self._until
    if active:
      for name in self._original:
        if name not in self._keep:
          self._reward_manager.get_term_cfg(name).weight = 0.0
    elif self._was_active:
      for name, weight in self._original.items():
        self._reward_manager.get_term_cfg(name).weight = weight
    self._was_active = active
    return {"active": torch.tensor(float(active))}
