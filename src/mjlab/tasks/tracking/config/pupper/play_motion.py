"""Replay a Pupper reference motion in simulation and say whether it holds up.

Physics replay (the default) drives the joints straight to the reference angles,
open loop, starting from the motion's first frame (root pose included), with
gravity and contacts on. If the robot tips over or drifts far from the
reference, plain replay won't work on the real robot either, and the motion is
a candidate for training with Mjlab-Tracking-Flat-Pupper-v3.

    # headless: print a verdict (good for coding agents)
    uv run python -m mjlab.tasks.tracking.config.pupper.play_motion \\
        src/mjlab/tasks/tracking/config/pupper/tricks/sit.npz

    # watch it: the robot plus a translucent ghost of the reference
    uv run python -m mjlab.tasks.tracking.config.pupper.play_motion motion.npz --viewer viser
    uv run python -m mjlab.tasks.tracking.config.pupper.play_motion motion.npz --viewer viser --share

    # the reference alone, no physics (the robot is placed on each frame)
    ... motion.npz --no-physics --viewer viser

    # a trained tracking policy instead of plain replay, scored the same way
    ... motion.npz --checkpoint logs/rsl_rl/pupper_tracking/<run>/model_3000.pt
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_ACTION_SCALE

# The robot's animation player plays tricks with these gains (animation_controller_py),
# stiffer than the walking policies' kp 5.5, so replay in sim uses them too.
REPLAY_KP = 7.5
REPLAY_KD = 0.25
# Verdict thresholds on the body's distance from the reference.
WOBBLE_POS_M, WOBBLE_TILT_DEG = 0.03, 10.0  # beyond this, replay wobbles
FALL_POS_M, FALL_TILT_DEG = 0.10, 45.0  # beyond this, the robot has fallen


def main(
  motion_file: tyro.conf.Positional[Path],
  viewer: Literal["none", "viser", "native"] = "none",
  share: bool = False,
  no_physics: bool = False,
  loops: int = 1,
  checkpoint: Path | None = None,
) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.tracking.config.pupper.env_cfgs import pupper_tracking_env_cfg
  from mjlab.tasks.tracking.mdp import MotionCommand

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = pupper_tracking_env_cfg(motion_file=str(motion_file.resolve()), play=True)
  cfg.scene.num_envs = 1
  cfg.terminations = {}  # watch the whole motion even if it goes wrong
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  actuator = articulation.actuators[0]
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  if checkpoint is None:  # a trained policy keeps the gains it was trained with
    actuator.stiffness, actuator.damping = REPLAY_KP, REPLAY_KD
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("motion")
  assert isinstance(cmd, MotionCommand)
  default = robot.data.default_joint_pos
  n_frames = int(cmd.motion.time_step_total)
  dt = env.step_dt

  class ReplayPolicy:
    """Joint targets = the reference's next frame; optionally teleport to the reference."""

    def __call__(self, obs) -> torch.Tensor:
      del obs
      nxt = torch.clamp(cmd.time_steps + 1, max=n_frames - 1)
      if no_physics:
        root = robot.data.default_root_state.clone()
        root[:, 0:3] = cmd.anchor_pos_w
        root[:, 3:7] = cmd.anchor_quat_w
        root[:, 7:] = 0.0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(cmd.joint_pos, torch.zeros_like(cmd.joint_pos))
      return (cmd.motion.joint_pos[nxt] - default) / PUPPER_ACTION_SCALE

  wrapped = RslRlVecEnvWrapper(env)
  policy = ReplayPolicy()
  if checkpoint is not None:
    from dataclasses import asdict

    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    rl_cfg = load_rl_cfg("Mjlab-Tracking-Flat-Pupper-v3")
    runner = MotionTrackingOnPolicyRunner(wrapped, asdict(rl_cfg), device=device)
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)

  if viewer != "none":
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    if viewer == "viser":
      import viser as viser_lib

      server = viser_lib.ViserServer(label="pupper-motion", share=share)
      if share:
        print("Open the viewer:", server.request_share_url())
      ViserPlayViewer(wrapped, policy, viser_server=server).run()
    else:
      NativeMujocoViewer(wrapped, policy).run()
    return

  wrapped.reset()
  obs = wrapped.get_observations()
  worst_pos, worst_tilt, fell_at, wobble_at = 0.0, 0.0, None, None
  timeline = []
  for step in range(n_frames * loops - 1):
    with torch.inference_mode():
      obs, *_ = wrapped.step(policy(obs))
    pos_err = float(torch.linalg.norm(cmd.robot_anchor_pos_w - cmd.anchor_pos_w))
    # angle between the robot's and the reference's body orientation
    dot = float(torch.abs((cmd.robot_anchor_quat_w * cmd.anchor_quat_w).sum()))
    tilt = math.degrees(2 * math.acos(min(1.0, dot)))
    worst_pos, worst_tilt = max(worst_pos, pos_err), max(worst_tilt, tilt)
    if (step + 1) % int(round(0.5 / dt)) == 0:
      timeline.append(f"{(step + 1) * dt:4.1f}s {pos_err * 100:5.1f}cm {tilt:5.1f}deg")
    if fell_at is None and (pos_err > FALL_POS_M or tilt > FALL_TILT_DEG):
      fell_at = (step + 1) * dt
    if wobble_at is None and (pos_err > WOBBLE_POS_M or tilt > WOBBLE_TILT_DEG):
      wobble_at = (step + 1) * dt
  mode = (
    f"trained policy {checkpoint.name}"
    if checkpoint
    else "kinematic (no physics)"
    if no_physics
    else "physics, open-loop joint replay"
  )
  print(f"motion        {motion_file.name}: {n_frames} frames, {n_frames * dt:.1f} s")
  print(f"mode          {mode}")
  print("error over time (every 0.5 s):")
  for line in timeline:
    print("   ", line)
  print(f"worst body position error   {worst_pos * 100:.1f} cm")
  print(f"worst body orientation error {worst_tilt:.1f} deg")
  train = (
    "uv run train Mjlab-Tracking-Flat-Pupper-v3 "
    f"--env.commands.motion.motion-file {motion_file}"
  )
  if fell_at is not None:
    print(
      f"verdict       FALLS at t = {fell_at:.2f} s. Plain replay won't work: make the "
      f"motion gentler, or train it:\n              {train}"
    )
  elif wobble_at is not None:
    print(
      f"verdict       WOBBLES from t = {wobble_at:.2f} s but stays up. Replay may work on "
      f"the robot; a trained policy will track it more tightly:\n              {train}"
    )
  else:
    who = "the policy" if checkpoint else "plain replay"
    print(f"verdict       HOLDS UP: {who} follows the reference closely.")
  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
