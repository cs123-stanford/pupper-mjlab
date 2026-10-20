"""The spider: its reference gait, its walking task, and its clip-trick module."""

from __future__ import annotations

import numpy as np

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  JOINT_NAMES,
  SPIDER_POSE,
  get_spec,
)
from mjlab.tasks.pupper_gait.mdp import gait_reference as gr
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  SPIDER_FREQUENCY,
  pupper_spider_walk_env_cfg,
)
from mjlab.tasks.tracking.config.pupper.tricks import spider


def test_spider_reference_trots_in_place_from_the_stance():
  table = gr.build_joint_reference_table(100, "spider")
  assert table.shape == (100, 12)
  model = get_spec().compile()
  limits = np.array([model.joint(n).range for n in JOINT_NAMES])
  assert np.all(table >= limits[:, 0] - 1e-9) and np.all(table <= limits[:, 1] + 1e-9)
  # Every leg spends two thirds of the cycle in the stance, and diagonal pairs
  # (FR + BL, FL + BR) lift together.
  at_stance = np.all(
    np.abs(table.reshape(100, 4, 3) - np.array(SPIDER_POSE).reshape(4, 3)) < 1e-9,
    axis=2,
  )
  assert np.allclose(at_stance.mean(axis=0), 2 / 3, atol=0.02)
  assert np.array_equal(at_stance[:, 0], at_stance[:, 3])
  assert np.array_equal(at_stance[:, 1], at_stance[:, 2])
  assert not np.array_equal(at_stance[:, 0], at_stance[:, 1])


def test_spider_walk_task_plays_the_spider_gait_from_its_stance():
  cfg = pupper_spider_walk_env_cfg()
  for params in (
    cfg.observations["actor"].terms["policy"].params,
    cfg.rewards["gait_tracking"].params,
  ):
    assert params["gait"] == "spider" and params["frequency"] == SPIDER_FREQUENCY
  joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
  assert joint_pos is not None
  assert list(joint_pos.values()) == list(SPIDER_POSE)


def test_spider_recording_and_stance_blend():
  rec = spider.recorded()
  assert abs(rec.duration - 11.3) < 0.1 and rec.joint_pos.shape[1] == 12
  settled = spider.settle_into_stance(rec)
  assert np.allclose(settled.joint_pos[-1], SPIDER_POSE)
  assert len(settled.joint_pos) == len(rec.joint_pos) + 30  # 0.6 s at 50 Hz
