"""Categorical latency buffers, mirroring the CS 123 MJX training pipeline.

The Pupper deploy stack has real latency in both directions: the IMU reading the
policy sees is up to one control step stale, and the motor target it emits arrives
up to one control step late. ``pupperv3-mjx`` -- the pipeline the deployable
walking policies were trained with -- randomizes both, and a policy trained
without them oscillates on hardware even though it walks in sim.

This is a direct port of ``pupperv3_mjx.utils.sample_lagged_value``: push the new
value to the front of a newest-first buffer, then read index ``lag`` where ``lag``
is drawn from a categorical distribution, **resampled independently every step and
per environment**. Element ``i`` of the distribution is the probability of an
``i``-step lag, so ``[0.2, 0.8]`` means "20% no lag, 80% one step of lag".

Note this is a per-step coin flip, not a lag sampled once per episode: the policy
must tolerate the latency changing underneath it, which is what the real
(jitter-prone) control loop does.

``mjlab``'s :class:`~mjlab.utils.buffers.DelayBuffer` samples its lag *uniformly*
over ``[min_lag, max_lag]``, so it can express the IMU distribution
(``[0.5, 0.5]`` is uniform over ``{0, 1}``) but not the action one
(``[0.2, 0.8]``). Both go through this buffer instead, so the two paths stay
symmetric and match the reference implementation exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

# Defaults from the CS 123 lab 5 training config (`training_config`):
#   latency_distribution     = [0.2, 0.8]  (action / motor command)
#   imu_latency_distribution = [0.5, 0.5]  (angular velocity + projected gravity)
PUPPER_ACTION_LATENCY_DIST: tuple[float, ...] = (0.2, 0.8)
PUPPER_IMU_LATENCY_DIST: tuple[float, ...] = (0.5, 0.5)

# Deterministic equivalent of the action distribution above, in *physics* steps.
#
# The robot's bus delay is fixed, not random: the deploy sim models it as
# ``command_latency_timesteps: 8`` at 520 Hz = 15.4 ms. That is 0.77 of a 20 ms
# control step, and ``[0.2, 0.8]`` is how MJX expresses a sub-step delay when the
# only buffer available is a whole-control-step one -- its mean, 0.8 steps = 16 ms,
# is the quantity that matches hardware. The per-step Bernoulli is a discretization
# artifact, not jitter the robot has.
#
# mjlab integrates at 4 ms, so it can hold the same 16 ms delay exactly: 4 physics
# steps, zero variance. That matters because the coin flip is invisible to the
# policy -- ``last_action`` is its own raw output -- so under the MJX model it can
# never learn the right phase lead and hedges between the two branches instead.
# The hedge is swamped by the reference while walking and dominates at low
# authority, which is where the residual stutter shows up.
PUPPER_ACTION_LATENCY_PHYSICS_STEPS: int = 4


class LatencyBuffer:
  """Newest-first value buffer with a per-step, per-env categorical lag.

  Args:
    distribution: Probability of each lag in steps; ``distribution[i]`` is the
      probability of reading the value from ``i`` steps ago. Must be non-empty
      and sum to a positive value (it is normalized internally).
    num_envs: Number of parallel environments.
    feature_shape: Trailing shape of the buffered value, e.g. ``(6,)``.
    device: Torch device for storage and sampling.
  """

  def __init__(
    self,
    distribution: Sequence[float],
    num_envs: int,
    feature_shape: tuple[int, ...],
    device: torch.device | str,
  ) -> None:
    if len(distribution) == 0:
      raise ValueError("Latency distribution must have at least one entry.")
    probs = torch.as_tensor(distribution, dtype=torch.float32, device=device)
    if torch.any(probs < 0.0) or float(probs.sum()) <= 0.0:
      raise ValueError(
        f"Latency distribution must be non-negative with positive sum, got "
        f"{list(distribution)}."
      )
    self._probs = (probs / probs.sum()).expand(num_envs, -1).contiguous()
    self._buffer = torch.zeros(
      (num_envs, len(distribution), *feature_shape), device=device
    )
    self._num_envs = num_envs

  @property
  def max_lag(self) -> int:
    """Largest lag the buffer can serve, in steps."""
    return self._buffer.shape[1] - 1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Zero the buffered history for the given environments.

    Matches the MJX env, which reinitializes the buffers to zeros on reset.
    """
    if env_ids is None:
      self._buffer.zero_()
    else:
      self._buffer[env_ids] = 0.0

  def step(self, value: torch.Tensor) -> torch.Tensor:
    """Push ``value`` and return a lagged sample.

    Args:
      value: Shape ``(num_envs, *feature_shape)``.

    Returns:
      The lagged value, same shape as ``value``.
    """
    # Roll then overwrite index 0: buffer[:, i] is the value from i steps ago.
    self._buffer = torch.roll(self._buffer, shifts=1, dims=1)
    self._buffer[:, 0] = value
    lag = torch.multinomial(self._probs, num_samples=1).squeeze(-1)
    return self._buffer[torch.arange(self._num_envs, device=value.device), lag]
