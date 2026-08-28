"""Export a trained Pupper trick (tracking) policy to the robot's policy JSON.

    uv run python -m mjlab.tasks.tracking.config.pupper.export_trick \\
        --checkpoint logs/rsl_rl/pupper_tracking/<run>/model_3000.pt \\
        --motion-file src/mjlab/tasks/tracking/config/pupper/tricks/sit.npz \\
        --name sit --out sit_policy.json

The JSON is the walking policies' deploy format (contract v2, see
``mjlab.tasks.pupper.export``) with a trick observation layout and two extra
blocks the robot needs to play the trick:

- ``trick``: name, frame rate and duration.
- ``motion``: the reference, one row per 50 Hz frame: joint angles and
  velocities (JOINT_NAMES order) and the body orientation (w, x, y, z). The
  robot steps through it while the policy runs, exactly as in training.

Observation frame (no history), in this order:
  motion_command       24  reference joint angles (12) and velocities (12) now
  motion_anchor_ori_b   6  reference body orientation relative to the robot's,
                           first two columns of the rotation matrix
  base_ang_vel          3  IMU angular velocity
  joint_pos_rel        12  joint angles minus the default pose
  joint_vel_rel        12  joint velocities
  last_action          12  the policy's previous output
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab

TRICK_OBS_COMPONENTS: tuple[tuple[str, int], ...] = (
  ("motion_command", 24),
  ("motion_anchor_ori_b", 6),
  ("base_ang_vel", 3),
  ("joint_pos_rel", 12),
  ("joint_vel_rel", 12),
  ("last_action", 12),
)
TRICK_OBS_DIM = sum(size for _, size in TRICK_OBS_COMPONENTS)


def export_trick_policy_from_env(
  actor: torch.nn.Module, env, name: str
) -> dict[str, Any]:
  """Build the trick's deploy JSON from a trained actor and its tracking env."""
  from mjlab.tasks.pupper.export import export_pupper_policy_from_env
  from mjlab.tasks.tracking.mdp import MotionCommand

  om = env.observation_manager
  dims = [d[0] for d in om.group_obs_term_dim["actor"]]
  if dims != [size for _, size in TRICK_OBS_COMPONENTS]:
    raise ValueError(
      f"actor observation {list(zip(om.active_terms['actor'], dims, strict=True))} does not "
      f"match the trick layout {TRICK_OBS_COMPONENTS}"
    )
  policy = export_pupper_policy_from_env(
    actor,
    env,
    single_obs_dim=TRICK_OBS_DIM,
    obs_components=TRICK_OBS_COMPONENTS,
    command_clip={},  # no velocity command in a trick
  )
  policy.pop("command_clip", None)

  cmd = env.command_manager.get_term("motion")
  assert isinstance(cmd, MotionCommand)
  motion = cmd.motion
  frames = int(motion.time_step_total)
  fps = round(1.0 / env.step_dt)
  anchor = cmd.motion_anchor_body_index

  def rows(t: torch.Tensor) -> list[list[float]]:
    return np.round(t.detach().cpu().numpy().astype(float), 6).tolist()

  policy["trick"] = {
    "name": name,
    "fps": fps,
    "frames": frames,
    "duration_s": frames / fps,
  }
  # Clip libraries (motion_design.concat_clips) carry their clip table, which the
  # robot's controller uses to play one clip at a time on request.
  from mjlab.tasks.tracking.config.pupper.motion_design import read_clips

  clips = read_clips(cmd.cfg.motion_file)
  if clips:
    policy["trick"]["clips"] = clips
  policy["motion"] = {
    "joint_pos": rows(motion.joint_pos),
    "joint_vel": rows(motion.joint_vel),
    "anchor_quat_wxyz": rows(motion.body_quat_w[:, anchor]),
  }
  return policy


def main(
  checkpoint: Path,
  motion_file: Path,
  name: str,
  out: Path,
) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.pupper.export import json_forward
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.tasks.tracking.config.pupper.env_cfgs import pupper_tracking_env_cfg
  from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = pupper_tracking_env_cfg(motion_file=str(motion_file.resolve()), play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  runner = MotionTrackingOnPolicyRunner(
    wrapped, asdict(load_rl_cfg("Mjlab-Tracking-Flat-Pupper-v3")), device=device
  )
  runner.load(str(checkpoint), map_location=device)
  actor = runner.alg.get_policy()
  policy = export_trick_policy_from_env(actor, env, name)

  # Parity: the exported network must reproduce the trained actor's mean action
  # on real observations from a short rollout.
  inference = runner.get_inference_policy(device=device)
  wrapped.reset()
  obs = wrapped.get_observations()
  worst = 0.0
  with torch.inference_mode():
    for _ in range(100):
      action = inference(obs)
      x = obs["actor"].detach().cpu().numpy()
      worst = max(
        worst, float(np.abs(json_forward(policy, x) - action.cpu().numpy()).max())
      )
      obs, *_ = wrapped.step(action)
  if worst > 1e-3:
    raise RuntimeError(
      f"exported network differs from the trained actor by {worst:.2e}"
    )

  out.write_text(json.dumps(policy))
  size_kb = out.stat().st_size / 1024
  print(
    f"wrote {out} ({size_kb:.0f} KB): trick '{name}', {policy['trick']['frames']} frames, "
    f"{policy['trick']['duration_s']:.1f} s; parity with the trained actor {worst:.1e}"
  )
  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
