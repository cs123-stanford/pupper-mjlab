"""Visualize a gait *reference* open-loop (no training needed).

Available gaits: ``trot`` (shipped) and ``lift`` (yours -- see the
TODO(student) block in ``mdp/gait_reference.py``; this refuses to run until
the block is filled in). Any other name plays the trot fallback, the same way
the tasks do (see the optional reference lab).

Drives the Pupper's joints directly to the phase-driven IK reference so you can
see the intended gait before training with it. Gravity is off by default so the
robot floats and the legs cycle cleanly; pass ``--gravity`` to see whether the
reference actually carries the robot, which is the check that matters before
trusting one.

Run (local native viewer):
    uv run python -m mjlab.tasks.pupper_gait.visualize_reference --gait lift

Colab / headless (viser share URL):
    uv run python -m mjlab.tasks.pupper_gait.visualize_reference --viser --share
"""

from __future__ import annotations

import tyro

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_ACTION_SCALE
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  GAIT_FREQUENCY,
  GAIT_GALLOP_FREQ_MULT,
)


def main(
  # VelocityFS on purpose: its env builds without any reference table, so the
  # trot can be inspected before the lift gait exists. The table shown comes
  # from --gait, not from the task.
  task: str = "Mjlab-VelocityFS-Flat-Pupper-v3",
  gait: str = "trot",
  frequency: float = GAIT_FREQUENCY,
  n_samples: int = 100,
  gravity: bool = False,
  viser: bool = False,
  share: bool = False,
) -> None:
  import torch

  import mjlab.tasks  # noqa: F401  (register tasks)
  from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import JOINT_NAMES
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.pupper_gait.mdp.gait import gait_phase
  from mjlab.tasks.pupper_gait.mdp.gait_reference import build_joint_reference_table
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  if not gravity:
    cfg.sim.mujoco.gravity = (0.0, 0.0, 0.0)

  base_env = ManagerBasedRlEnv(cfg=cfg, device=device)
  robot = base_env.scene["robot"]
  default = robot.data.default_joint_pos  # (1, 12)

  table = torch.tensor(
    build_joint_reference_table(n_samples, gait), device=device, dtype=torch.float32
  )
  # Match the env: the gallop reference cycles GAIT_GALLOP_FREQ_MULT x faster.
  # Each gait runs at the cadence it is actually used at: the two gallops cycle
  # GAIT_GALLOP_FREQ_MULT x faster, the fast MixedGaits reference uses its own
  # multiplier, and the walking gaits use the base frequency. Read the multiplier
  # from the table rather than hardcoding it, so the viewer cannot drift out of
  # step with what the tasks actually run.
  from mjlab.tasks.pupper_gait.mdp.gait_reference import _GAIT_FREQ_MULT

  mult = _GAIT_FREQ_MULT.get(
    gait, GAIT_GALLOP_FREQ_MULT if gait in ("gallop", "cheetah") else 1.0
  )
  eff_frequency = frequency * mult
  print(
    f"{gait} reference table: {tuple(table.shape)} joints={list(JOINT_NAMES)} "
    f"cycle={1.0 / eff_frequency:.3f}s"
  )

  class ReferencePolicy:
    """Return the action that drives joints to the current gait reference."""

    def __call__(self, obs) -> torch.Tensor:
      del obs
      x = gait_phase(base_env, eff_frequency) * n_samples
      i0 = torch.floor(x).long() % n_samples
      i1 = (i0 + 1) % n_samples
      frac = (x - torch.floor(x)).unsqueeze(1)
      reference = table[i0] * (1.0 - frac) + table[i1] * frac
      return (reference - default) / PUPPER_ACTION_SCALE

  policy = ReferencePolicy()
  env = RslRlVecEnvWrapper(base_env, clip_actions=load_rl_cfg(task).clip_actions)
  if viser:
    import viser as viser_lib

    server = viser_lib.ViserServer(label="pupper-gait", share=share)
    if share:
      print("Open the viewer:", server.request_share_url())
    ViserPlayViewer(env, policy, viser_server=server).run()
  else:
    NativeMujocoViewer(env, policy).run()


if __name__ == "__main__":
  tyro.cli(main)
