"""Per-command gait selection: a different reference for each kind of motion.

The single triangular trot reference is a forward-walking pattern, so it fights
every command that is not forward walking. Turning in place still asks the feet to
stride fore-aft; running fast asks for a gait the robot cannot reach; running
backward asks for the trot played in reverse, which is not how this robot moves
backward fastest. This picks a reference per environment from the command:

===============  ==========================================================
mode             reference
===============  ==========================================================
slow fore/aft    ``trot`` -- the original triangular gait, unchanged. Below the
                 gallop onset the robot is walking, and the proven reference is
                 the right one.
fast forward     ``reach`` -- the emergent +1.5 m/s gait captured from a
                 trained policy (see ``gait_reference._CAPTURED_GAITS_FILE``),
                 roll-symmetrized.
fast backward    ``reach_back`` -- the emergent -1.5 m/s gait from the same
                 policy, likewise symmetrized. A separate capture rather than
                 the forward gait time-reversed: how this robot actually moves
                 backward fast is not the forward gait played in reverse.
turn / sidestep  ``lift`` -- diagonal pairs lifting and setting down in place,
                 no fore-aft travel, so the reference does not oppose the
                 commanded rotation or lateral motion.
===============  ==========================================================

Only the fast gait crouches, which is also why ``base_height`` needs no special
handling here: it already relaxes above the same onset.

The mode is chosen from the *command*, which is piecewise constant between
resamples, so it cannot chatter within a command window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper.mdp.observations import PupperObsNoise, pupper_actor_obs
from mjlab.tasks.pupper_gait.mdp.gait import _DIR_THRESHOLD, _lookup, _reference_table
from mjlab.tasks.pupper_gait.mdp.gait_reference import _GAIT_FREQ_MULT
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_DEFAULT_NOISE = PupperObsNoise()


def mixed_reference_offset(
  env: "ManagerBasedRlEnv",
  command_name: str,
  frequency: float,
  blend_speed: float,
  n_samples: int,
  gallop_speed: float,
) -> torch.Tensor:
  """``(N, 12)`` reference offset: the extended-reach trot, or lift-in-place."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  vx, vy, yaw = command[:, 0], command[:, 1], command[:, 2]

  robot: Entity = env.scene["robot"]
  default = robot.data.default_joint_pos
  assert default is not None

  translating = vx.abs() >= _DIR_THRESHOLD

  # The fast gait runs a quicker phase clock (see _GAIT_FREQ_MULT); the walking
  # and turning references share the base cadence.
  fast_mult = _GAIT_FREQ_MULT.get("reach", 1.0)
  freq = frequency * torch.where(
    translating & (vx.abs() >= gallop_speed),
    torch.full_like(vx, fast_mult),
    torch.ones_like(vx),
  )
  phase = torch.frac(env.episode_length_buf.float() * env.step_dt * freq)
  # Time-reverse for backward / negative-turn commands. torch.frac keeps the sign;
  # the floor-modulo in _lookup wraps it.
  dir_signal = torch.where(translating, vx, yaw)
  eff_phase = torch.where(dir_signal >= 0, phase, torch.frac(-phase))

  trot = _lookup(_reference_table(env, n_samples, "trot"), eff_phase, n_samples)
  # The fast gait is direction-split: forward and backward are separate
  # captures of how the robot actually moves at speed in each direction. The
  # reach_back table ships pre-reversed (see gait_reference), so the shared
  # backward time-reversal below plays the capture forward as recorded.
  reach = _lookup(_reference_table(env, n_samples, "reach"), eff_phase, n_samples)
  reach_back = _lookup(
    _reference_table(env, n_samples, "reach_back"), eff_phase, n_samples
  )
  lift = _lookup(_reference_table(env, n_samples, "lift"), eff_phase, n_samples)
  fast = translating & (vx.abs() >= gallop_speed)
  fast_ref = torch.where((vx >= 0).unsqueeze(1), reach, reach_back)
  gait_ref = torch.where(translating.unsqueeze(1), trot, lift)
  gait_ref = torch.where(fast.unsqueeze(1), fast_ref, gait_ref)

  offset = gait_ref - default
  speed = torch.linalg.vector_norm(
    torch.stack([vx, vy, yaw], dim=-1), dim=-1, keepdim=True
  )
  blend = torch.clamp(speed / blend_speed, 0.0, 1.0)

  return blend * offset


class mixed_gait_tracking:
  """``exp(-mean(error^2 / (std * std_scale)^2))`` against the per-command reference.

  Same shape as :class:`~mjlab.tasks.pupper_gait.mdp.gait.gait_tracking`, but the
  tolerance is scaled per environment so the backward-fast mode can opt out.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.ids = cfg.params["asset_cfg"].joint_ids
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"], list_of_strings=joint_names
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)
    self._p = {
      k: cfg.params[k]
      for k in (
        "command_name",
        "frequency",
        "blend_speed",
        "n_samples",
        "gallop_speed",
        "phase_lead_steps",
      )
    }
    for gait in ("trot", "reach", "reach_back", "lift"):
      _reference_table(env, int(self._p["n_samples"]), gait)  # warm the caches

  def __call__(self, env: "ManagerBasedRlEnv", **kwargs) -> torch.Tensor:
    del kwargs
    asset: Entity = env.scene["robot"]
    default = asset.data.default_joint_pos
    assert default is not None
    lead = float(self._p["phase_lead_steps"])
    # Rewind the clock by the pipeline delay, as the single-gait reward does.
    saved = env.episode_length_buf
    env.episode_length_buf = (saved.float() - lead).to(saved.dtype)
    try:
      offset = mixed_reference_offset(
        env,
        str(self._p["command_name"]),
        float(self._p["frequency"]),
        float(self._p["blend_speed"]),
        int(self._p["n_samples"]),
        float(self._p["gallop_speed"]),
      )
    finally:
      env.episode_length_buf = saved
    reference = (default + offset)[:, self.ids]
    error_squared = torch.square(asset.data.joint_pos[:, self.ids] - reference)
    return torch.exp(-torch.mean(error_squared / self.std**2, dim=1))


class roll_asymmetry:
  """Sustained left-right leg-roll asymmetry, as an EMA of the pair sums.

  Leg roll is **motor 2**: perturbing the FK shows ``_2`` is the joint that
  moves the foot in/out laterally (``_1`` swings it fore-aft, ``_3`` is the
  knee). A mirrored pose negates ``_2`` (the trot reference's left column is
  the negated right column), so a symmetric gait keeps ``roll_R + roll_L``
  zero-mean over a cycle, while the failure mode observed in the al7sdood
  capture -- one leg tucked in and its partner splayed out, pair sums of
  0.44-0.68 rad sustained -- survives averaging. An EMA with time constant
  ``tau_s`` (~one fast gait cycle) cancels the legitimate within-cycle
  oscillation and keeps the DC lean; the returned value is the mean over the
  front and back pairs of the squared EMA. Use with a strongly negative
  weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene["robot"]
    pairs = (
      ("leg_front_r_2", "leg_front_l_2"),
      ("leg_back_r_2", "leg_back_l_2"),
    )
    ids = []
    for right, left in pairs:
      for name in (right, left):
        jid, jname = asset.find_joints((name,))
        assert jname == [name], f"roll (motor 2) joint {name} not found"
        ids.append(jid[0])
    self._ids = torch.tensor(ids, device=env.device).view(2, 2)
    tau_s = float(cfg.params.get("tau_s", 0.3))
    self._alpha = env.step_dt / (tau_s + env.step_dt)
    self._ema = torch.zeros(env.num_envs, 2, device=env.device)

  def __call__(self, env: "ManagerBasedRlEnv", **kwargs) -> torch.Tensor:
    del kwargs
    jp = env.scene["robot"].data.joint_pos
    pair_sums = jp[:, self._ids].sum(dim=2)  # (N, 2): front, back
    # A fresh episode seeds the EMA from the current pose rather than decaying
    # the previous episode's residue.
    fresh = (env.episode_length_buf <= 1).unsqueeze(1)
    updated = self._ema + self._alpha * (pair_sums - self._ema)
    self._ema = torch.where(fresh, pair_sums, updated)
    return (self._ema**2).mean(dim=1)


def mixed_gait_actor_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  frequency: float,
  blend_speed: float,
  n_samples: int,
  gallop_speed: float,
  add_noise: bool = True,
  noise: PupperObsNoise = _DEFAULT_NOISE,
  asset_cfg: SceneEntityCfg = _ROBOT,
  imu_latency_distribution=None,
) -> torch.Tensor:
  """36-dim proprio frame + the 12-dim per-command reference offset = 48."""
  base = pupper_actor_obs(
    env, command_name, add_noise, noise, asset_cfg, imu_latency_distribution
  )
  offset = mixed_reference_offset(
    env,
    command_name,
    frequency,
    blend_speed,
    n_samples,
    gallop_speed,
  )
  return torch.cat([base, offset], dim=-1)


def zero_reference_actor_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  frequency: float,
  blend_speed: float,
  n_samples: int,
  gallop_speed: float,
  zero_reference: bool = True,
  add_noise: bool = True,
  noise: PupperObsNoise = _DEFAULT_NOISE,
  asset_cfg: SceneEntityCfg = _ROBOT,
  imu_latency_distribution=None,
) -> torch.Tensor:
  """The 48-dim gait frame with an empty reference: proprio + 12 zero dims.

  Same architecture as the mixed-gait tasks, so checkpoints and the deploy
  path are interchangeable, but the policy learns plain velocity tracking with
  nothing in the reference slot. The phase-clock parameters are carried (and
  ignored) so the exporter recognizes the frame family; ``zero_reference`` is
  the marker it dispatches on, exporting constant default-pose tables whose
  on-robot offset is identically zero -- exactly what training showed.
  """
  del frequency, blend_speed, n_samples, gallop_speed, zero_reference
  base = pupper_actor_obs(
    env, command_name, add_noise, noise, asset_cfg, imu_latency_distribution
  )
  return torch.cat([base, base.new_zeros(base.shape[0], 12)], dim=-1)


def mixed_gait_reference_obs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  frequency: float,
  blend_speed: float,
  n_samples: int,
  gallop_speed: float,
) -> torch.Tensor:
  """Just the ``(N, 12)`` reference offset, for the critic.

  The actor already carries this inside its 48-dim frame; the critic only needs the
  reference itself, not another copy of the proprioception it already has.
  """
  offset = mixed_reference_offset(
    env,
    command_name,
    frequency,
    blend_speed,
    n_samples,
    gallop_speed,
  )
  return offset
