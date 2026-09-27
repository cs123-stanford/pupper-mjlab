"""Pupper v3 motion tracking (BeyondMimic) tasks.

    uv run train Mjlab-Tracking-Flat-Pupper-v3 --env.scene.num-envs 4096 \
        --env.commands.motion.motion-file path/to/your_trick.npz

Without a motion file it trains the worked example, tricks/sit.npz. Design
motions with motion_design.py (see tricks/sit.py).
"""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pupper_tracking_env_cfg
from .rl_cfg import pupper_tracking_ppo_runner_cfg

DEFAULT_MOTION = str(Path(__file__).resolve().parent / "tricks" / "sit.npz")

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pupper-v3",
  env_cfg=pupper_tracking_env_cfg(motion_file=DEFAULT_MOTION),
  play_env_cfg=pupper_tracking_env_cfg(motion_file=DEFAULT_MOTION, play=True),
  rl_cfg=pupper_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
