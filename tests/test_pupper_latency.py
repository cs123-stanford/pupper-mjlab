"""Tests for the Pupper deploy-latency randomization.

The Pupper deploy stack lags both directions -- the IMU reading is up to one
control step stale and the motor target arrives up to one step late -- and a
policy trained without that oscillates on hardware while walking cleanly in a
latency-free sim. These tests pin the two properties that make the port faithful:
the lag distribution matches ``pupperv3-mjx``, and the lag reaches only the things
it reaches on the robot.
"""

import pytest
import torch

from mjlab.tasks.pupper.mdp.latency import (
  PUPPER_ACTION_LATENCY_DIST,
  PUPPER_IMU_LATENCY_DIST,
  LatencyBuffer,
)


def _buffer(dist, num_envs=20000, features=1):
  return LatencyBuffer(dist, num_envs, (features,), "cpu")


def test_lag_distribution_matches_mjx():
  """``distribution[i]`` is the probability of an i-step lag, per env, per step."""
  torch.manual_seed(0)
  for dist in (PUPPER_ACTION_LATENCY_DIST, PUPPER_IMU_LATENCY_DIST):
    buf = _buffer(dist)
    buf.step(torch.zeros(20000, 1))  # t=0: lag-1 slot holds zeros
    out = buf.step(torch.ones(20000, 1))  # t=1: lag-0 -> 1.0, lag-1 -> 0.0
    p_lag0 = (out[:, 0] == 1.0).float().mean().item()
    assert p_lag0 == pytest.approx(dist[0], abs=0.02)


def test_lag_resampled_every_step_not_fixed_per_episode():
  """MJX draws a fresh lag each step; a per-episode lag would be a different DR."""
  torch.manual_seed(0)
  buf = _buffer((0.5, 0.5), num_envs=4096)
  buf.step(torch.zeros(4096, 1))
  # Feed a distinct value each step; count how often an env's lag flips.
  prev = (buf.step(torch.full((4096, 1), 1.0))[:, 0] == 1.0).float()
  flips = 0.0
  for v in range(2, 12):
    cur = (buf.step(torch.full((4096, 1), float(v)))[:, 0] == float(v)).float()
    flips += (cur != prev).float().mean().item()
    prev = cur
  # Independent 50/50 draws flip ~50% of the time; a fixed lag would never flip.
  assert flips / 10 == pytest.approx(0.5, abs=0.05)


def test_reset_clears_history_per_env():
  buf = _buffer((0.0, 1.0), num_envs=4)  # always lag 1, so we read the history
  buf.step(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
  buf.reset(torch.tensor([0, 2]))
  out = buf.step(torch.zeros(4, 1))
  assert out.flatten().tolist() == [0.0, 2.0, 0.0, 4.0]


def test_zero_lag_distribution_is_passthrough():
  buf = _buffer((1.0,))
  assert buf.max_lag == 0
  value = torch.randn(20000, 1)
  assert torch.equal(buf.step(value), value)


@pytest.mark.parametrize("dist", [(), (-1.0, 2.0), (0.0, 0.0)])
def test_invalid_distribution_rejected(dist):
  with pytest.raises(ValueError):
    _buffer(dist)


@pytest.mark.slow
def test_latency_reaches_imu_but_not_last_action():
  """The deploy contract: the IMU lags, ``last_action`` does not.

  On the robot the controller feeds back its own emitted action (measured equal
  to the previous policy output to 1.4e-6), and the joint encoders are read
  straight through -- only the IMU is buffered. The observation frame layout must
  also be unchanged, since the export folds it by fixed offsets.
  """
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  torch.manual_seed(0)
  cfg = load_env_cfg("Mjlab-VelocityFS-Flat-Pupper-v3", play=True)
  cfg.scene.num_envs = 128
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (128, 20 * 48)

    stale = 0
    prev_imu = None
    for _ in range(60):
      action = torch.randn(env.num_envs, 12) * 0.3
      actor_obs = env.step(action)[0]["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      frame = actor_obs.view(env.num_envs, 20, 48)[:, -1]

      # last_action is the undelayed raw action (only per-subgroup noise differs).
      alive = env.episode_length_buf > 0  # a reset zeroes raw actions
      err = (frame[:, 24:36] - action).abs().max(dim=1).values
      assert err[alive].max() <= 0.011

      if prev_imu is not None:
        stale += (
          ((frame[:, :6] - prev_imu).abs().max(dim=1).values < 1e-12).float().mean()
        )
      prev_imu = frame[:, :6]

    # A served frame repeats the previous one when lag goes 0 -> 1: p = 0.5*0.5.
    assert stale / 59 == pytest.approx(0.25, abs=0.06)
  finally:
    env.close()


def test_deterministic_latency_is_the_default_and_does_not_leak():
  """The default model is a fixed actuator delay, not the MJX Bernoulli.

  Both express the same 16 ms mean, but only the deterministic one has zero
  variance, which is what lets the policy learn an exact phase lead. Also pins
  that building a task does not mutate the shared module-level actuator config --
  the delay is applied by replacing it, and assigning through would leak into
  every task built afterwards, including the ones that want the MJX model.
  """
  import mjlab.tasks  # noqa: F401
  from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
    PUPPER_ACTUATOR,
    PUPPER_DECIMATION,
  )
  from mjlab.envs.mdp.actions import JointPositionActionCfg
  from mjlab.tasks.pupper.mdp.latency import PUPPER_ACTION_LATENCY_PHYSICS_STEPS
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-VelocityFS-Flat-Pupper-v3")
  assert cfg.scene.entities["robot"].articulation is not None
  act = cfg.scene.entities["robot"].articulation.actuators[0]

  n = PUPPER_ACTION_LATENCY_PHYSICS_STEPS
  assert act.delay_min_lag == act.delay_max_lag == n, "delay must be deterministic"
  # 4 physics steps at the Pupper's 4 ms dt = 16 ms, against a 15.4 ms real bus,
  # and strictly under one 20 ms control step.
  assert 0 < n < PUPPER_DECIMATION

  # The env-step delay term is not also applied; double-delaying would silently
  # double the lag.
  assert type(cfg.actions["joint_pos"]) is JointPositionActionCfg
  # No hidden lag for the critic to be told about.
  assert "applied_action" not in cfg.observations["critic"].terms

  assert PUPPER_ACTUATOR.delay_min_lag == 0, "leaked delay into the shared actuator"
  assert PUPPER_ACTUATOR.delay_max_lag == 0, "leaked delay into the shared actuator"


@pytest.mark.slow
def test_critic_sees_applied_action_actor_does_not():
  """Under the MJX latency model, the applied target is privileged: critic only.

  Without it the value function cannot see the plant's actual input, since
  ``last_action`` reports the policy's raw output and the two disagree whenever
  the sampled lag is nonzero. The actor must not gain it -- the robot cannot
  observe its own bus lag, and the deploy frame layout is fixed.
  """
  import mjlab.tasks  # noqa: F401
  import mjlab.tasks.pupper.pupper_env_cfg as pupper_env_cfg
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.pupper.mdp.actions import DelayedJointPositionAction
  from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
    pupper_velocity_fs_env_cfg,
  )

  torch.manual_seed(0)
  # The registry caches env configs at import, so rebuild under the MJX model
  # rather than loading the registered (deterministic) one.
  previous = pupper_env_cfg.PUPPER_LATENCY_MODEL
  pupper_env_cfg.PUPPER_LATENCY_MODEL = "mjx"
  try:
    cfg = pupper_velocity_fs_env_cfg()
  finally:
    pupper_env_cfg.PUPPER_LATENCY_MODEL = previous
  cfg.scene.num_envs = 256
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    assert "applied_action" in env.observation_manager.active_terms["critic"]
    # Actor stays the deployable 48-dim frame x 20 history.
    actor_obs = env.observation_manager.compute()["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (256, 20 * 48)

    term = env.action_manager.get_term("joint_pos")
    assert isinstance(term, DelayedJointPositionAction)
    lagged = 0
    for _ in range(40):
      env.step(torch.randn(env.num_envs, 12) * 0.3)
      applied, processed = term.applied_actions, term._processed_actions
      lagged += float((~torch.isclose(applied, processed).all(dim=1)).float().mean())
    # Matches PUPPER_ACTION_LATENCY_DIST = (0.2, 0.8).
    assert lagged / 40 == pytest.approx(0.8, abs=0.06)
  finally:
    env.close()


def test_applied_action_rejects_undelayed_term():
  """Fail loudly rather than silently duplicating ``last_action``."""
  from mjlab.envs.mdp.actions.actions import JointPositionAction
  from mjlab.tasks.pupper.mdp.observations import applied_action

  class _Env:
    class action_manager:
      @staticmethod
      def get_term(name):
        del name
        return object.__new__(JointPositionAction)

  with pytest.raises(TypeError, match="no command latency"):
    applied_action(_Env(), "joint_pos")  # type: ignore[arg-type]


@pytest.mark.slow
def test_airborne_pretrain_isolates_gait_tracking_then_releases():
  """Phase one trains the motion alone; phase two restores the full objective.

  Also pins the ordering requirement: pretrain_rewards must be the last
  curriculum term -- a curriculum that rewrites weights on every compute would
  otherwise undo the zeroing mid-warmup.

  Skips until the lift gait's TODO(student) block in ``mdp/gait_reference.py``
  is filled in: building a StableGait env needs the lift reference table.
  """
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
    GAIT_PRETRAIN_HEIGHT,
    GAIT_PRETRAIN_ITERS,
    GAIT_STEPS_PER_ITER,
  )
  from mjlab.tasks.registry import load_env_cfg

  torch.manual_seed(0)
  cfg = load_env_cfg("Mjlab-StableGait-Flat-Pupper-v3")
  assert list(cfg.curriculum)[-1] == "pretrain_rewards"
  # Every weight ships at zero; set a few the way the lab notebook does, so
  # the warmup has something to isolate and release.
  cfg.rewards["gait_tracking"].weight = 5.0
  cfg.rewards["track_linear_velocity"].weight = 10.0
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.scene.num_envs = 32
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  except NotImplementedError as e:
    pytest.skip(str(e))
  try:
    env.reset()
    rm = env.reward_manager
    for _ in range(20):
      env.step(torch.randn(env.num_envs, 12) * 0.4)

    live = {n for n in rm.active_terms if rm.get_term_cfg(n).weight != 0.0}
    assert live == {"gait_tracking"}, f"warmup should isolate the gait, got {live}"
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    assert torch.allclose(
      height, torch.full_like(height, GAIT_PRETRAIN_HEIGHT), atol=0.02
    )

    # Releasing restores every weight the config set, not just the kept one.
    env.common_step_counter = GAIT_PRETRAIN_ITERS * GAIT_STEPS_PER_ITER
    env.curriculum_manager.compute()
    after = {n for n in rm.active_terms if rm.get_term_cfg(n).weight != 0.0}
    assert after > live, "releasing the warmup must restore the other terms"
    assert "track_angular_velocity" in after
  finally:
    env.close()
