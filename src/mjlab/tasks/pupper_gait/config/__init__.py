from mjlab.tasks.pupper.rl import PupperOnPolicyRunner
from mjlab.tasks.pupper_gait.mystery_env_cfg import (
  pupper_mystery_bumpy_env_cfg,
  pupper_mystery_env_cfg,
)
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  pupper_mixed_gaits_bumpy_env_cfg,
  pupper_mixed_gaits_env_cfg,
  pupper_stable_gait_bumpy_env_cfg,
  pupper_stable_gait_env_cfg,
  pupper_velocity_fs_env_cfg,
)
from mjlab.tasks.pupper_gait.rl_cfg import pupper_gait_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task

# Every task here ships with every reward weight at zero -- the terms and what
# they measure are given, but choosing which ones matter, their signs, and
# their magnitudes is your job (see the lab notebook, which passes your
# weights into training).

# VelocityFS ("from scratch"): the plain flat velocity objective in the
# gait-frame architecture -- 48-dim actor frame with the reference dims pinned
# to zero. Trains like the velocity task, deploys (and swaps checkpoints)
# like the gait tasks.
_velocity_fs_rl = pupper_gait_ppo_runner_cfg()
_velocity_fs_rl.experiment_name = "pupper_velocity_fs"
register_mjlab_task(
  task_id="Mjlab-VelocityFS-Flat-Pupper-v3",
  env_cfg=pupper_velocity_fs_env_cfg(),
  play_env_cfg=pupper_velocity_fs_env_cfg(play=True),
  rl_cfg=_velocity_fs_rl,
  runner_cls=PupperOnPolicyRunner,
)

_stable_rl = pupper_gait_ppo_runner_cfg()
_stable_rl.experiment_name = "pupper_stable_gait"
register_mjlab_task(
  task_id="Mjlab-StableGait-Flat-Pupper-v3",
  env_cfg=pupper_stable_gait_env_cfg(),
  play_env_cfg=pupper_stable_gait_env_cfg(play=True),
  rl_cfg=_stable_rl,
  runner_cls=PupperOnPolicyRunner,
)

# StableGait on Perlin bumps. Same experiment dir as the flat task so a
# converged flat checkpoint warm-starts the bumpy run.
register_mjlab_task(
  task_id="Mjlab-StableGait-Bumpy-Pupper-v3",
  env_cfg=pupper_stable_gait_bumpy_env_cfg(),
  play_env_cfg=pupper_stable_gait_bumpy_env_cfg(play=True),
  rl_cfg=_stable_rl,
  runner_cls=PupperOnPolicyRunner,
)

# Per-command mixed references: trot for fore/aft, lift-in-place for
# turn/sidestep, and a separate fast branch above the gallop onset. What plays
# in the fast branch is up to you -- out of the box it is the trot, and finding
# a reference that beats it is the optional lab (see mdp.gait_reference).
_mixed_rl = pupper_gait_ppo_runner_cfg()
_mixed_rl.experiment_name = "pupper_mixed_gaits"
register_mjlab_task(
  task_id="Mjlab-MixedGaits-Flat-Pupper-v3",
  env_cfg=pupper_mixed_gaits_env_cfg(),
  play_env_cfg=pupper_mixed_gaits_env_cfg(play=True),
  rl_cfg=_mixed_rl,
  runner_cls=PupperOnPolicyRunner,
)

# MixedGaits on Perlin bumps; same experiment dir so flat checkpoints
# warm-start it.
register_mjlab_task(
  task_id="Mjlab-MixedGaits-Bumpy-Pupper-v3",
  env_cfg=pupper_mixed_gaits_bumpy_env_cfg(),
  play_env_cfg=pupper_mixed_gaits_bumpy_env_cfg(play=True),
  rl_cfg=_mixed_rl,
  runner_cls=PupperOnPolicyRunner,
)

# ???
_mystery_rl = pupper_gait_ppo_runner_cfg()
_mystery_rl.experiment_name = "pupper_mystery"
register_mjlab_task(
  task_id="Mjlab-Mystery-Flat-Pupper-v3",
  env_cfg=pupper_mystery_env_cfg(),
  play_env_cfg=pupper_mystery_env_cfg(play=True),
  rl_cfg=_mystery_rl,
  runner_cls=PupperOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Mystery-Bumpy-Pupper-v3",
  env_cfg=pupper_mystery_bumpy_env_cfg(),
  play_env_cfg=pupper_mystery_bumpy_env_cfg(play=True),
  rl_cfg=_mystery_rl,
  runner_cls=PupperOnPolicyRunner,
)
