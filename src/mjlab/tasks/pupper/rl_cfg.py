"""RL configuration for the Pupper v3 velocity task.

The actor architecture matches the CS 123 notebook: a 4-hidden-layer ELU MLP with
observation normalization (the normalization is later folded into the first layer
by the deploy exporter). The critic can be arbitrary since it is never deployed.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pupper_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 128, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.97,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pupper_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def pupper_max_speed_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Separate experiment dir so the max-speed study does not mix with velocity runs."""
  cfg = pupper_ppo_runner_cfg()
  cfg.experiment_name = "pupper_max_speed"
  return cfg
