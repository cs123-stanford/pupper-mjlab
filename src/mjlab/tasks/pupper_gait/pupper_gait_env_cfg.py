"""Pupper v3 trot-gait task.

Reuses the Pupper velocity env (movement, DR, knee/height penalties, critic) but
swaps the static standing-pose reward for a phase-driven trot reference: the
policy is rewarded for matching a diagonal-trot joint trajectory while moving,
and observes the gait phase so it can sync to it.
"""

from __future__ import annotations

import math

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import PUPPER_DECIMATION
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper.pupper_env_cfg import (
  pupper_flat_env_cfg,
)
from mjlab.tasks.pupper_gait import mdp
from mjlab.terrains.config import perlin_noise
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

# Bumpy-only terrain: smooth Gaussian-like Perlin bumps (like the MJX random
# height field), no stairs/slopes. Height scales with per-patch difficulty up to
# ~5 cm -- noticeable for a ~14 cm-tall Pupper but walkable. Patches are sampled
# at random difficulty (curriculum=False), so the policy sees the full bump range
# from the start.
BUMPY_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=5.0,
  num_rows=10,
  num_cols=20,
  curriculum=False,
  sub_terrains={"bumpy": perlin_noise(proportion=1.0, height_range=(0.0, 0.05))},
  add_lights=True,
)

# Gait cycle time [s] and the corresponding phase frequency [Hz]: one full
# swing/stance cycle per leg.
GAIT_CYCLE_S: float = 0.75
GAIT_FREQUENCY: float = 1.0 / GAIT_CYCLE_S

# Command magnitude at which the reference reaches the full trot; below it the
# reference blends toward the static default pose (stand at zero command).
GAIT_BLEND_SPEED: float = 0.1

# Reference-table resolution (phase samples over one cycle).
GAIT_N_SAMPLES: int = 100

# Commanded |vx| at/above which the reference switches from the diagonal trot to
# the gallop (and height tracking is neglected).
#
# DISABLED: set above GAIT_MAX_LIN_VEL_X so no command ever selects the gallop,
# leaving one trot reference across the whole speed range. The gallop reference is
# not trackable by this robot -- it demands 1.44 rad on a knee in a single 20 ms
# control step, which needs kp * 1.44 = 7.9 Nm against a 3.0 Nm effort limit. The
# trot's worst case is 0.37 rad / 2.05 Nm, which is why the trot imitates cleanly
# and the gallop does not. Sweeping stride x cadence finds nothing both
# actuator-feasible and fast enough to be worth a second gait: every feasible
# setting tops out near 0.25 m/s, below the 0.5 m/s that used to select it. The
# binding constraint is that the 0.08 m gallop stride works the leg near full
# extension, where the IK is ill-conditioned (1% of gallop samples exceed 1 cm
# residual, vs 0% for the trot).
#
# Everything below is left wired up so this is a one-line revival. Inspect a
# candidate before trusting it, with gravity ON -- floating, a bad reference looks
# fine:
#   uv run python -m mjlab.tasks.pupper_gait.visualize_reference --gait gallop --gravity
GAIT_GALLOP_SPEED: float = 1.0e3

# Height-tracking std while galloping: large enough to effectively neglect height
# at top speed, so the body can pitch/bob through the gallop.
GAIT_GALLOP_HEIGHT_STD: float = 10.0

# Phase-clock speedup while galloping: the gallop reference cycles this much
# faster than the trot (a quicker cadence, since it only fires at high speed).
GAIT_GALLOP_FREQ_MULT: float = 2

# Max forward/backward command [m/s] for the trot tasks; lateral (y) and yaw
# ranges are left as the velocity task set them. The old 1.0 existed only because
# the gallop was meant to cover the top of the range. With the gallop split into
# its own task this is back near the velocity task's 0.75 -- and the trot
# reference's no-slip foot travel only implies ~0.2 m/s, so commanding much beyond
# this sets track_linear_velocity against gait_tracking.
GAIT_MAX_LIN_VEL_X: float = 0.7

# Forward command range [m/s] for the dedicated gallop task. Forward-only: the
# reference is a rotatory footfall order with a lead, so it is not symmetric under
# time reversal the way the diagonal trot is.
GALLOP_LIN_VEL_X_RANGE: tuple[float, float] = (0.0, 1.0)

# ``gallop_speed`` value that makes the gallop the *only* reference: the branch is
# ``|vx| >= gallop_speed``, so 0.0 selects it for every command including zero.
GALLOP_ALWAYS: float = 0.0

# Speed-switched trot/gallop tasks: trot up to TROTGALLOP_ONSET, gallop above it,
# with the command range extended to cover the gallop band. Kept as separate task
# ids rather than re-enabling the gallop on the trot tasks, so the trot results
# stay comparable across runs.
#
# 0.5 m/s is where the trot starts to come apart in practice, which is also the
# threshold the gallop originally used. The trot reference's no-slip foot travel
# only implies ~0.2 m/s, so by 0.5 the feet are already sliding through stance.
TROTGALLOP_ONSET: float = 0.5
TROTGALLOP_MAX_LIN_VEL_X: float = 1.0

# MixedGaits-specific overrides (the TrotGallop tasks share _trot_gallop, so
# these are applied on top of it rather than through the shared constants).
#
# Commands out to +-1.5 m/s: whatever plays in the fast slot has to cover this
# band (the shipped trot fallback will not enjoy it -- see the optional
# reference lab).
MIXED_MAX_LIN_VEL_X: float = 1.5
# The Pupper tasks terminate at 30 deg of body tilt (the notebook's
# terminal_body_angle, tuned for the trot). A committed fast reach gait
# legitimately pitches harder than a trot, so 30 deg cut episodes exactly when
# the policy switched to the right gait at speed, teaching it to avoid the
# fast band. 45 deg still catches real falls (the base velocity task uses 70).
MIXED_FELL_OVER_LIMIT_ANGLE: float = math.radians(45.0)

# StableGait: the mixed-gaits per-command selection without the fast reach
# branch. The triangular trot covers forward AND backward across the whole
# command range, and the lift-in-place cycle covers turning/sidestepping.
# Commands cap at 0.7 m/s, and the phase clock runs quicker than the 1.33 Hz
# base cadence: the trot's no-slip travel scales with cadence, so a faster
# cycle adapts the fixed stride to the top of the range instead of switching to
# the reach gait.
#
# 1.5 Hz rather than the 2.0 the task shipped with: at 2.0 (runs g6ugoiq2,
# 1mc78ipo) backward/turn/sidestep were smooth but forward was not, and at
# 50 Hz control a 2.0 Hz cycle leaves only ~7 control steps per swing to place
# a foot. 1.5 Hz gives ~11, still 1.125x the base cadence's no-slip travel.
STABLE_GAIT_FREQUENCY: float = 1.5
STABLE_GAIT_MAX_LIN_VEL_X: float = 0.7

GAIT_TRACK_ANG_VEL_STD: float = 0.7071  # the shared task's value, sqrt(0.5).

# Airborne pretraining: for this many PPO iterations the base is pinned upright and
# stationary in the air and every reward but gait_tracking is zeroed, so the policy
# learns the reference motion before it has to also learn balance. See mdp.warmup.
GAIT_PRETRAIN_ITERS: int = 500
# Base height while pinned [m]. The reference puts the feet 0.14 m below the base
# at stance, so this leaves ~0.16 m of clearance and the legs never touch down.
GAIT_PRETRAIN_HEIGHT: float = 0.30

# Per-joint tolerance for tracking the reference pose. The knee (_3) and hip (_2)
# move through the trot; abduction (_1) stays near default, so it is held tightest.
GAIT_STD = {r".*_1": 0.15, r".*_2": 0.3, r".*_3": 0.3}

# Much looser tolerance for the gallop task: the reference is a *suggestion* there,
# not a target to hit, so the policy is free to find its own fast gait while being
# nudged toward the gallop shape. Velocity tracking drives the behavior instead.
#
# Calibrated against the do-nothing baseline: the raw gait_tracking a policy gets
# for emitting zero actions while the reference cycles. Headroom (1 - baseline) is
# what tracking the gallop can actually earn, so it sets how hard the term pulls.
# Measured in the live env over 40 steps at 128 envs -- estimating it from the
# reference amplitude alone overstates the error, since the robot does not sit at
# the default pose and low-speed commands blend the reference away:
#
#   std     do-nothing   headroom (x the gait_tracking weight)
#   trot      0.318      0.682   too tight for an untrackable reference
#   2x        0.720      0.280   <- here: real but secondary to velocity
#   3.3x      0.881      0.119   weak
#   5x        0.944      0.056   vestigial
#
# Kept a uniform multiple of the trot std rather than scaling per joint by how much
# each moves in the reference: the hip barely moves in the gallop, so a
# motion-proportional std would clamp it *tighter* than the trot does and forbid
# exactly the freestyling this is meant to allow.
GALLOP_STD = {r".*_1": 0.3, r".*_2": 0.6, r".*_3": 0.6}

# Command-side heading hold, mirroring the deploy stack's IMU yaw P-loop (the
# same correction the G1 carry tasks train with): walking straight, heading is
# neutrally stable (ang-vel rewards damp only the instantaneous yaw rate) and
# wanders over a run. A blind heading-deviation reward cannot fix this -- neither
# actor nor critic can observe the held target, so it only adds advantage noise
# (this replaced exactly such a penalty) -- but closing the loop through the yaw
# command works even on a frozen yaw-rate-tracking policy. Active in play and
# training alike, so policies train on the same closed-loop command profile
# deploy produces. KEEP KP/CLIP IN SYNC WITH THE DEPLOY-SIDE CONTROLLER
# (pupper_gait_deploy's neural_controller); the exporter stamps these into the
# deploy JSON's ``heading_hold`` block.
GAIT_HEADING_HOLD_KP: float = 1.0
GAIT_HEADING_HOLD_CLIP: float = 0.3

# Iterations are converted to env steps via GAIT_STEPS_PER_ITER;
# common_step_counter advances once per env step, so a curriculum threshold is
# iterations x num_steps_per_env.
GAIT_STEPS_PER_ITER: int = 24

# Yaw-rate command curriculum: (PPO iteration at which the stage starts, maximum
# |yaw| [rad/s]). The range is symmetric, so 1.0 means (-1.0, 1.0). Yaw tracking
# learns but slowly against the full +/-2 rad/s from step 0, so establish the turn
# on easy commands first and widen from there.
#
# This deliberately re-adds a ``command_vel`` curriculum that ``_add_gait`` used to
# pop outright. The inherited one ramped *every* axis, which held the linear
# commands back and slowed yaw learning; these stages touch yaw only, leaving
# lin_vel_x/y at the ranges _add_gait sets.
GAIT_YAW_STAGES: tuple[tuple[int, float], ...] = (
  (0, 1.0),
  (5_000, 1.5),
  (10_000, 2.0),
)


# How far to rewind the phase clock for the *reward* only, in control steps.
#
# The policy observes ref(k), but episode_length_buf increments before the reward
# is computed, so it is graded against ref(k+1); and its action does not reach the
# actuators for another PUPPER_ACTION_LATENCY_PHYSICS_STEPS / PUPPER_DECIMATION of
# a step. The gap is the sum, and without this the policy has to spend capacity
# learning a constant feedforward lead just to break even -- measured at ~20% of
# gait_tracking for a policy that does not compensate.
#
# Free at deploy: a constant phase shift of a periodic gait is unobservable, and
# the reward does not exist on the robot. The *observation* keeps a zero lead,
# since the on-robot controller has to reproduce it exactly.
#
# Raises the absolute value of gait_tracking for identical behavior, so reward
# curves are not comparable to runs from before this change.
GAIT_REWARD_PHASE_LEAD_STEPS: float = 1.0 + (
  mdp.PUPPER_ACTION_LATENCY_PHYSICS_STEPS / PUPPER_DECIMATION
)


def _gait_params(gallop_speed: float = GAIT_GALLOP_SPEED) -> dict:
  return {
    "command_name": "twist",
    "frequency": GAIT_FREQUENCY,
    "blend_speed": GAIT_BLEND_SPEED,
    "n_samples": GAIT_N_SAMPLES,
    "gallop_speed": gallop_speed,
    "gallop_freq_mult": GAIT_GALLOP_FREQ_MULT,
  }


def _add_gait(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
  gallop_speed: float = GAIT_GALLOP_SPEED,
  lin_vel_x_range: tuple[float, float] = (-GAIT_MAX_LIN_VEL_X, GAIT_MAX_LIN_VEL_X),
  std: dict[str, float] | None = None,
  yaw_stages: tuple[tuple[int, float], ...] | None = GAIT_YAW_STAGES,
  shared_critic: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Swap the static posture reward for phase-driven reference tracking.

  ``gallop_speed`` is the ``|vx|`` at which the reference switches from trot to
  gallop; it must be threaded identically through the observation, the reward and
  the height term, or the policy would observe one reference and be graded on
  another. Pass ``GALLOP_ALWAYS`` for a gallop-only task.
  """
  gait_params = _gait_params(gallop_speed)
  std = GAIT_STD if std is None else std
  obs_params = {
    # Noise and IMU latency stay on in play, matching the velocity task and
    # the MJX pipeline's eval env (see pupper_env_cfg for why).
    "add_noise": True,
    "imu_latency_distribution": mdp.PUPPER_IMU_LATENCY_DIST,
    **gait_params,
  }

  def _proprio_group() -> ObservationGroupCfg:
    # Proprio frame (incl. velocity command) + reference pose motion the policy
    # tracks (36 + 12 = 48), 20-step history.
    return ObservationGroupCfg(
      terms={
        "policy": ObservationTermCfg(
          # With a shared critic both groups evaluate this, so route through the
          # per-step cache -- the IMU latency buffer is env-shared and would
          # otherwise be advanced twice per step. See mdp.gait.
          func=mdp.pupper_gait_shared_obs
          if shared_critic
          else mdp.pupper_gait_actor_obs,
          params=dict(obs_params),
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=20,
    )

  cfg.observations["actor"] = _proprio_group()
  if shared_critic:
    # Symmetric actor-critic, as in pupperv3-mjx: one observation feeds both
    # networks, and it is the fully corrupted one (noise + IMU lag). Every
    # privileged term is dropped, including gait_reference -- the reference is
    # already inside the 48-dim frame.
    cfg.observations["critic"] = _proprio_group()
    cfg.events["reset_gait_shared_obs"] = EventTermCfg(
      mode="reset", func=mdp.reset_gait_shared_obs, params={}
    )
  else:
    # Critic also sees the reference offset (privileged value estimation).
    cfg.observations["critic"].terms["gait_reference"] = ObservationTermCfg(
      func=mdp.gait_reference_obs, params=dict(gait_params)
    )

  # Replace the static posture reward with reference-pose tracking (which is the
  # static default at zero command and the trot at speed). Zero weight like every
  # term -- your call whether (and how hard) the reference gets tracked.
  cfg.rewards.pop("pose", None)
  cfg.rewards["gait_tracking"] = RewardTermCfg(
    func=mdp.gait_tracking,
    weight=0.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "std": std,
      # Reward only -- see GAIT_REWARD_PHASE_LEAD_STEPS. The actor observation
      # deliberately keeps a zero lead so the robot can reproduce it.
      "phase_lead_steps": GAIT_REWARD_PHASE_LEAD_STEPS,
      **gait_params,
    },
  )

  # Roll/pitch-rate stabilizer. Added rather than replacing track_yaw_velocity,
  # which stays the clean yaw signal (this one folds roll/pitch rate into its
  # error, which is exactly what makes it useful as a damper and useless as the
  # yaw objective -- see mdp.track_yaw_velocity).
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_angular_velocity,
    weight=0.0,
    params={"command_name": "twist", "std": GAIT_TRACK_ANG_VEL_STD},
  )

  # Neglect height tracking while galloping (std widens to GAIT_GALLOP_HEIGHT_STD).
  if "base_height" in cfg.rewards:
    bh = cfg.rewards["base_height"]
    bh.func = mdp.gait_base_height
    bh.params = {
      "target_height": bh.params["target_height"],
      "std": bh.params["std"],
      "gallop_std": GAIT_GALLOP_HEIGHT_STD,
      "gallop_speed": gallop_speed,
      "command_name": "twist",
    }

  # Forward/backward command range; lateral range is left as the velocity task set
  # it. Yaw starts at the first curriculum stage rather than its full range, so the
  # initial config matches what the curriculum applies at step 0.
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = lin_vel_x_range

  # Command-side heading hold (see GAIT_HEADING_HOLD_KP): walking straight, the
  # emitted yaw command becomes a clipped P-correction toward the heading
  # captured when the yaw command went quiet, exactly as the deploy controller
  # does with the IMU yaw. Replaces the old blind heading_deviation penalty.
  twist.heading_hold_kp = GAIT_HEADING_HOLD_KP
  twist.heading_hold_clip = GAIT_HEADING_HOLD_CLIP

  # Replace the inherited all-axis command curriculum with yaw-only stages (see
  # GAIT_YAW_STAGES). ``yaw_stages=None`` skips the ramp and opens the full range
  # from step 0. Play has no curriculum manager, so it always gets the full range
  # outright -- otherwise eval would run at the stage-0 yaw range forever.
  cfg.curriculum.pop("command_vel", None)
  full_yaw = GAIT_YAW_STAGES[-1][1]
  if play or yaw_stages is None:
    twist.ranges.ang_vel_z = (-full_yaw, full_yaw)
  else:
    twist.ranges.ang_vel_z = (-yaw_stages[0][1], yaw_stages[0][1])
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": iters * GAIT_STEPS_PER_ITER, "ang_vel_z": (-mag, mag)}
          for iters, mag in yaw_stages
        ],
      },
    )

  # Airborne pretraining (see GAIT_PRETRAIN_ITERS / mdp.warmup). Skipped in play,
  # which has no curriculum manager -- the pin would then never release.
  #
  # Registered LAST on purpose: the curriculum manager runs terms in insertion
  # order and the ramps above rewrite their own weights every compute, so a term
  # running after this one would undo the zeroing mid-warmup.
  if not play:
    pretrain_steps = GAIT_PRETRAIN_ITERS * GAIT_STEPS_PER_ITER
    cfg.events["hold_base_airborne"] = EventTermCfg(
      mode="step",
      func=mdp.hold_base_airborne,
      params={"height": GAIT_PRETRAIN_HEIGHT, "until_step": pretrain_steps},
    )
    cfg.curriculum["pretrain_rewards"] = CurriculumTermCfg(
      func=mdp.pretrain_rewards,
      params={"keep": ("gait_tracking",), "until_step": pretrain_steps},
    )
  return cfg


def _apply_mixed_reference(
  cfg: ManagerBasedRlEnvCfg, frequency: float, gallop_speed: float
) -> ManagerBasedRlEnvCfg:
  """Swap the single-gait observation/reward for the per-command mixed reference.

  ``gallop_speed`` is the ``|vx|`` at which the fast reach gait takes over;
  passing ``GAIT_GALLOP_SPEED`` disables it, leaving trot for fore/aft and lift
  for turn/sidestep only.
  """
  # The mixed reference has no gallop branch and no per-mode tolerance, so it takes
  # only the phase-clock parameters.
  params = {
    "command_name": "twist",
    "frequency": frequency,
    "blend_speed": GAIT_BLEND_SPEED,
    "n_samples": GAIT_N_SAMPLES,
    "gallop_speed": gallop_speed,
  }

  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "policy": ObservationTermCfg(
        func=mdp.mixed_gait_actor_obs,
        params={
          "add_noise": True,
          "imu_latency_distribution": mdp.PUPPER_IMU_LATENCY_DIST,
          **params,
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=20,
  )
  cfg.observations["critic"].terms["gait_reference"] = ObservationTermCfg(
    func=mdp.mixed_gait_reference_obs, params=dict(params)
  )

  # base_height needs no adjustment: only the fast reference crouches, and the term
  # already relaxes above the same onset that selects it.

  cfg.rewards["gait_tracking"] = RewardTermCfg(
    func=mdp.mixed_gait_tracking,
    weight=0.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "std": GAIT_STD,
      "phase_lead_steps": GAIT_REWARD_PHASE_LEAD_STEPS,
      **params,
    },
  )
  return cfg


def pupper_mixed_gaits_env_cfg(
  play: bool = False, base: ManagerBasedRlEnvCfg | None = None
) -> ManagerBasedRlEnvCfg:
  """Flat ground with a per-command gait reference (see ``mdp.mixed_gaits``).

  The trot below the gallop onset, direction-split captured gaits above it --
  the roll-symmetrized emergent +-1.5 m/s gaits of run ``al7sdood`` (see
  ``mdp.gait_reference._CAPTURED_GAITS_FILE``) -- and a lift-in-place cycle
  for turning and sidestepping. Commands run out to +-1.5 m/s, the fall-over
  termination is loosened to 45 deg, velocity and gait tracking are raised,
  and a severe penalty holds the abduction pairs mirror-symmetric -- see the
  ``MIXED_*`` constants.

  Deployable: the exporter ships the captured forward gait in the fast slot,
  the backward capture as ``gallop_back_table``, and the lift table; the
  controller plays each exactly as training does (the backward table ships
  pre-reversed so the shared phase reversal cancels). Needs the
  ``pupper_gait_deploy`` controller with direction-split fast-table support.
  """
  cfg = _trot_gallop(base if base is not None else pupper_flat_env_cfg(play=play), play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-MIXED_MAX_LIN_VEL_X, MIXED_MAX_LIN_VEL_X)
  cfg.terminations["fell_over"].params["limit_angle"] = MIXED_FELL_OVER_LIMIT_ANGLE
  cfg = _apply_mixed_reference(cfg, GAIT_FREQUENCY, TROTGALLOP_ONSET)
  # Anti-lean penalty on the abduction pairs (a fast gait's classic failure
  # mode is one leg tucked in, its partner splayed out); zeroed during airborne
  # pretraining like every non-tracking term (see mdp.pretrain_rewards).
  cfg.rewards["roll_asymmetry"] = RewardTermCfg(
    func=mdp.roll_asymmetry,
    weight=0.0,
    params={"tau_s": 0.3},
  )
  return cfg


def pupper_mixed_gaits_bumpy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """MixedGaits on the Perlin bump field (see ``_make_bumpy``)."""
  return _make_bumpy(pupper_mixed_gaits_env_cfg(play=play))


def pupper_stable_gait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """MixedGaits without the fast reach branch, capped at 0.7 m/s (see constants).

  The triangular trot covers forward and backward (time-reversed) across the
  whole command range, the lift-in-place cycle covers turning and sidestepping,
  and both run on a quicker phase clock so the fixed trot stride slips less at the
  top of the range. No gait crouches, so ``base_height`` tracks tightly at every
  speed.

  Deployable: the on-robot controller plays the exported ``lift_table`` when
  the command is not translating and gates the (inert) fast table above the
  never-crossed ``gallop_speed``, so the exported reference reproduces this
  task's exactly.
  """
  cfg = _add_gait(
    pupper_flat_env_cfg(play=play),
    play,
    lin_vel_x_range=(-STABLE_GAIT_MAX_LIN_VEL_X, STABLE_GAIT_MAX_LIN_VEL_X),
    yaw_stages=None,
  )
  # No curriculum except the airborne gait pretrain (hold_base_airborne +
  # pretrain_rewards, the "learn the gait without gravity" phase).
  # GAIT_GALLOP_SPEED sits above the command range, so the fast branch never
  # fires: trot for fore/aft, lift for turn/sidestep.
  cfg = _apply_mixed_reference(cfg, STABLE_GAIT_FREQUENCY, GAIT_GALLOP_SPEED)
  return cfg


def pupper_stable_gait_bumpy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """StableGait on the Perlin bump field (see ``_make_bumpy``).

  Same task, rougher ground: random-difficulty bumps up to ~5 cm and the flat
  task's critic, so a flat StableGait checkpoint warm-starts this cleanly.
  """
  return _make_bumpy(pupper_stable_gait_env_cfg(play=play))


def pupper_velocity_fs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The from-scratch flat velocity task in the gait-frame architecture.

  Rewards, commands, and terminations are exactly the flat velocity task's;
  only the actor frame changes: the 48-dim gait layout with the reference
  component pinned to zero (see ``mdp.zero_reference_actor_obs``). Policies
  trained here deploy through the same path as the gait policies -- the
  exporter ships constant default-pose tables, whose on-robot offset is
  identically zero -- and share their input shape, so they drop into any slot
  the gait policies fit.
  """
  cfg = pupper_flat_env_cfg(play=play)

  # Fixed command ranges, MixedGaits-style: vx capped at +-0.7, vy/yaw as the
  # gait tasks train them, and the same command-side heading hold so deploy
  # (which stamps the hold from the export) matches training. NO curriculum:
  # the base task inherits the upstream all-axis command_vel ramp, whose later
  # stages widen vx to (-1.5, 2.0) at ~5k iterations and (-2.0, 3.0) at ~10k
  # -- untrackable for a Pupper, and the cause of run 2yazp80j's collapse
  # (err_xy 0.09 -> 0.65 at exactly that boundary). The exported command_clip
  # now matches what training actually sampled.
  twist = cfg.commands["twist"]
  assert isinstance(twist, mdp.UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-0.7, 0.7)
  twist.ranges.lin_vel_y = (-0.5, 0.5)
  twist.ranges.ang_vel_z = (-2.0, 2.0)
  twist.heading_hold_kp = GAIT_HEADING_HOLD_KP
  twist.heading_hold_clip = GAIT_HEADING_HOLD_CLIP
  cfg.curriculum = {}
  params = {
    "command_name": "twist",
    "frequency": GAIT_FREQUENCY,
    "blend_speed": GAIT_BLEND_SPEED,
    "n_samples": GAIT_N_SAMPLES,
    "gallop_speed": GAIT_GALLOP_SPEED,
  }
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "policy": ObservationTermCfg(
        func=mdp.zero_reference_actor_obs,
        params={
          "add_noise": True,
          "imu_latency_distribution": mdp.PUPPER_IMU_LATENCY_DIST,
          "zero_reference": True,
          **params,
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=20,
  )
  return cfg


def _make_bumpy(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Swap in the Perlin bump field and match the flat tasks' critic."""
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = BUMPY_TERRAINS_CFG
  # Bumpy-only, no difficulty progression: patches are random (see cfg above).
  cfg.curriculum.pop("terrain_levels", None)

  # Match the flat task's critic (drop the terrain height-scan + its raycaster) so
  # the observation is identical to the flat gait task -- this lets a flat-trained
  # trot+gallop checkpoint warm-start the bumpy run cleanly (the actor is already
  # identical; without this the critic input grows 84 -> 271 and the load fails).
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  return cfg


def _trot_gallop(cfg: ManagerBasedRlEnvCfg, play: bool) -> ManagerBasedRlEnvCfg:
  """Trot below ``TROTGALLOP_ONSET``, gallop above it, commands out to 1.0 m/s.

  Runs without the yaw curriculum: the full yaw range is live from step 0. The
  curriculum was introduced to work around yaw tracking that would not climb, and
  that turned out to be the shared velocity term folding roll/pitch rate into the
  yaw error rather than anything about the command schedule (see
  ``mdp.track_yaw_velocity``). With the reward corrected the scaffolding is worth
  removing -- it is also what let the gallop learn, since a gallop wobbles far
  more than a trot and the old term taxed it hardest.
  """
  return _add_gait(
    cfg,
    play,
    gallop_speed=TROTGALLOP_ONSET,
    lin_vel_x_range=(-TROTGALLOP_MAX_LIN_VEL_X, TROTGALLOP_MAX_LIN_VEL_X),
    yaw_stages=None,
  )
