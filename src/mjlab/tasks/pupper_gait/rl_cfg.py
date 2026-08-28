"""RL configs for the Pupper gait tasks (same net as velocity, own log dirs)."""

from mjlab.tasks.pupper.rl_cfg import pupper_ppo_runner_cfg

# A generous default budget; the lab notebook sets its own max_iterations.
_GAIT_MAX_ITERATIONS = 20_000


def pupper_gait_ppo_runner_cfg():
  cfg = pupper_ppo_runner_cfg()
  cfg.experiment_name = "pupper_gait"
  cfg.max_iterations = _GAIT_MAX_ITERATIONS
  return cfg


def pupper_gallop_ppo_runner_cfg():
  """Separate experiment dir so gallop runs do not land among the trot runs."""
  cfg = pupper_ppo_runner_cfg()
  cfg.experiment_name = "pupper_gallop"
  cfg.max_iterations = _GAIT_MAX_ITERATIONS
  return cfg


def pupper_trot_gallop_ppo_runner_cfg():
  """Separate experiment dir for the speed-switched trot/gallop tasks."""
  cfg = pupper_ppo_runner_cfg()
  cfg.experiment_name = "pupper_trot_gallop"
  cfg.max_iterations = _GAIT_MAX_ITERATIONS
  return cfg
