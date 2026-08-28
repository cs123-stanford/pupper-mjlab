"""The mystery task.

What does it train? The task id will not tell you, and neither will this
docstring. It is MixedGaits with something extra: an extra reference slot the
schedule opens now and then, reward machinery that pays for something no other
task pays for, and -- out of the box -- only the trot playing in that slot,
which earns almost none of it. The reference the slot is *waiting for* is
yours to design or capture (see the optional reference lab in
:mod:`mdp.gait_reference`). Get it right and you will know.
"""

from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pupper_gait import mdp
from mjlab.tasks.pupper_gait.mdp.gait_reference import jump_capture_duration_s
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  GAIT_BLEND_SPEED,
  GAIT_FREQUENCY,
  GAIT_N_SAMPLES,
  GAIT_REWARD_PHASE_LEAD_STEPS,
  GAIT_STD,
  GALLOP_STD,
  TROTGALLOP_ONSET,
  pupper_mixed_gaits_env_cfg,
)

# The jump-start grid: one BASE gait cycle (0.75 s). v1 used the universal
# 1.5 s boundary that every gait clock shares, because the slot had to both
# enter and exit on a boundary. With both edges cross-faded and the
# reference window down to ~0.8 s, exits land mid-cycle anyway -- so starts
# only need the base-cycle gap, halving trigger-to-jump lag (max 0.75 s,
# mean ~0.4). Fast-gait envs (2.5x clock) enter mid-cycle through the same
# fade that already handles every exit.
JUMP_SLOT_S: float = 1.0 / GAIT_FREQUENCY

# Standing height and the slot-motion geometry constants.
JUMP_STAND_HEIGHT: float = 0.145
JUMP_HEIGHT_CAP: float = 0.40
JUMP_TORSO_CLEARANCE: float = 0.02

# A captured slot motion plays over its real duration; without one (the
# shipped default -- supplying a better slot reference is the optional lab)
# the slot plays the trot for one base cycle and these windows derive from
# that.
_capture = jump_capture_duration_s()
JUMP_PLAYBACK_S: float = _capture if _capture is not None else JUMP_SLOT_S

# The jump's ACTIVE window inside a slot: the playback plus the exit fade,
# after which the reference stitches straight back into the flowing gait --
# the stride IS the landing recovery. The v1 window held the landing crouch
# for the rest of the 1.5 s grid unit, and the policy learned exactly what
# that rewarded: killing its momentum to land stationary, so jumps carried no
# velocity. Slot STARTS stay on the JUMP_SLOT_S grid; only
# the regime window shortened, and the exit cross-fade absorbs resuming
# mid-cycle.
JUMP_ACTIVE_S: float = 0.0  # set below once the capture duration is known

# Velocity tracking earns double inside the jump window (in-function
# multiplier on the windowed term): carrying the commanded velocity through
# the air is the window's headline objective.
JUMP_SLOT_VEL_MULT: float = 2.0

# Schedule sampling (see mdp.reset_jump_schedule): 15% of episodes carry no
# jump so pure locomotion stays in distribution; the rest get one to three
# slots, at least two grid units apart -- more airtime per batch for the
# carry-through gradient.
JUMP_P_NO_JUMP: float = 0.15

# Linear cross-fade at each window edge (3 control steps): zero exactly at
# the boundaries, so the composite reference is continuous by construction --
# the raw swap otherwise snaps 0.2-0.5 rad in one step at the worst
# boundaries.
JUMP_CROSS_FADE_S: float = 0.06

# Touchdown lands ~0.84 s into the capture (liftoff at 0.30 -- the window
# started 0.30 s before the source episode's liftoff -- plus the measured
# 0.54 s flight). The REFERENCE window ends a lead before that: the fade to
# the running gait completes ~0.1 s pre-touchdown, so the legs are already
# cycling for the stride when the feet meet the floor -- run, jump, run.
JUMP_TOUCHDOWN_S: float = 0.84 if _capture is not None else JUMP_PLAYBACK_S
JUMP_PRE_LAND_LEAD_S: float = 0.10
JUMP_ACTIVE_S = JUMP_TOUCHDOWN_S - JUMP_PRE_LAND_LEAD_S + JUMP_CROSS_FADE_S

# The REWARD window stays the full playback: the robot is airborne past the
# reference stitch, and flipping the masked terms back mid-fall would punish
# ballistics (lin_vel_z_l2 on a 2 m/s descent) and bill a stride the feet
# cannot yet take. Loose tracking std, the jump ladder, the velocity boost
# and the off-in-slot masks all key on this longer window.
JUMP_REWARD_WINDOW_S = JUMP_PLAYBACK_S + JUMP_CROSS_FADE_S
assert JUMP_ACTIVE_S < JUMP_REWARD_WINDOW_S
# The reward window outlives a grid unit now; the schedule's min separation
# (and request_jump's busy check) keep windows disjoint.
assert JUMP_REWARD_WINDOW_S < 2.0 * JUMP_SLOT_S


# Widened sim2real DR for the robustness finetune. The base Pupper env
# already randomizes these (friction 0.6-1.4, kp scale 0.6-1.1, kd scale
# 0.8-1.5); the finetune stretches them so the deployed policy has seen
# slicker floors and weaker/soggier motors than the real robot will show it
# -- the drift after a hard testing day was exactly "the motors got weaker".
ROBUST_FOOT_FRICTION_RANGE: tuple[float, float] = (0.35, 1.6)
ROBUST_KP_RANGE: tuple[float, float] = (0.5, 1.2)
ROBUST_KD_RANGE: tuple[float, float] = (0.6, 1.8)


def pupper_mystery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The mystery task (see module docstring)."""
  return _apply_jump_slots(pupper_mixed_gaits_env_cfg(play=play), play)


def pupper_mystery_bumpy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The mystery task on Perlin bumps, with widened sim2real randomization.

  Built on the rough scene, then ``_make_bumpy`` (which also drops the
  critic's height scan so flat-trained checkpoints warm-start with identical
  observation shapes), then the jump layer, then the widened friction and
  PD-gain randomization. Intended to be resumed from a converged flat
  MixedGaitsJump checkpoint rather than trained from scratch.
  """
  from mjlab.tasks.pupper.pupper_env_cfg import pupper_rough_env_cfg
  from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import _make_bumpy

  cfg = pupper_mixed_gaits_env_cfg(play=play, base=pupper_rough_env_cfg(play=play))
  cfg = _make_bumpy(cfg)
  cfg = _apply_jump_slots(cfg, play)

  cfg.events["foot_friction"].params["ranges"] = ROBUST_FOOT_FRICTION_RANGE
  cfg.events["pd_gains"].params["kp_range"] = ROBUST_KP_RANGE
  cfg.events["pd_gains"].params["kd_range"] = ROBUST_KD_RANGE

  # This is a finetune-only task, resumed from a converged flat checkpoint:
  # the airborne imitation pretrain would pin a mature policy in the air and
  # zero every reward for its first 500 iterations, delaying the actual
  # robustness objective. Dropped here; the flat task keeps it.
  cfg.events.pop("hold_base_airborne", None)
  cfg.curriculum.pop("pretrain_rewards", None)
  return cfg


def _apply_jump_slots(cfg: ManagerBasedRlEnvCfg, play: bool) -> ManagerBasedRlEnvCfg:
  """Layer the jump-slot machinery onto a MixedGaits cfg."""

  params = {
    "command_name": "twist",
    "frequency": GAIT_FREQUENCY,
    "blend_speed": GAIT_BLEND_SPEED,
    "n_samples": GAIT_N_SAMPLES,
    "gallop_speed": TROTGALLOP_ONSET,
    "jump_duration_s": JUMP_PLAYBACK_S,
    "slot_s": JUMP_ACTIVE_S,
    "cross_fade_s": JUMP_CROSS_FADE_S,
  }

  # Composite reference in the same 48-dim actor frame / privileged critic
  # layout as MixedGaits, so checkpoints stay shape-compatible.
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "policy": ObservationTermCfg(
        func=mdp.mixed_jump_actor_obs,
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
    func=mdp.mixed_jump_reference_obs, params=dict(params)
  )

  # One tracking term over the composite reference: strict locomotion std,
  # loose inside the slot. Keeps the "gait_tracking" name so the inherited
  # airborne pretrain's keep-list still matches.
  weight = cfg.rewards["gait_tracking"].weight
  cfg.rewards["gait_tracking"] = RewardTermCfg(
    func=mdp.mixed_jump_tracking,
    weight=weight,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "std": GAIT_STD,
      "std_slot": GALLOP_STD,
      "std_loose_s": JUMP_REWARD_WINDOW_S,
      "phase_lead_steps": GAIT_REWARD_PHASE_LEAD_STEPS,
      **params,
    },
  )

  # Velocity carry-through: stock linear tracking outside the slot, xy-only
  # inside (the shared term's vz^2 folding would zero the gradient in flight).
  # Weight and std carry over from MixedGaits.
  lin = cfg.rewards["track_linear_velocity"]
  lin.func = mdp.track_linear_velocity_windowed
  lin.params = {
    "std": lin.params["std"],
    "command_name": "twist",
    "slot_s": JUMP_REWARD_WINDOW_S,
    "slot_weight_mult": JUMP_SLOT_VEL_MULT,
  }

  # Terms that fight a jump: off inside the slot, stock outside (per-step
  # mask; weights unchanged). base_height keeps its gait_base_height params.
  for name, wrapper in (
    ("base_height", mdp.base_height_outside_slot),
    ("lin_vel_z_l2", mdp.lin_vel_z_l2_outside_slot),
    ("air_time", mdp.air_time_outside_slot),
    ("foot_slip", mdp.foot_slip_outside_slot),
    ("stand_still_pose", mdp.stand_still_pose_outside_slot),
    ("stand_still_joint_velocity", mdp.stand_still_joint_velocity_outside_slot),
  ):
    assert name in cfg.rewards, f"expected '{name}' in the MixedGaits reward set"
    cfg.rewards[name].func = wrapper
    cfg.rewards[name].params = {
      "slot_s": JUMP_REWARD_WINDOW_S,
      **cfg.rewards[name].params,
    }

  # The reward machinery no other task pays for. Slot-gated; what it is worth
  # -- if anything -- is your call, like every weight in this codebase.
  cfg.rewards["jump_up_velocity"] = RewardTermCfg(
    func=mdp.slot_jump_up_velocity,
    weight=0.0,
    params={"slot_s": JUMP_REWARD_WINDOW_S},
  )
  cfg.rewards["jump_airborne_height"] = RewardTermCfg(
    func=mdp.slot_jump_airborne_height,
    weight=0.0,
    params={
      "slot_s": JUMP_REWARD_WINDOW_S,
      "stand_height": JUMP_STAND_HEIGHT,
      "height_cap": JUMP_HEIGHT_CAP,
    },
  )

  # Belly no-hit zone, always on -- landing a jump belly-first bumps the
  # battery whether or not a slot is active, and locomotion gets the guard
  # for free.
  cfg.rewards["torso_clearance"] = RewardTermCfg(
    func=mdp.torso_clearance,
    weight=0.0,
    params={"min_clearance": JUMP_TORSO_CLEARANCE},
  )

  # The per-episode jump schedule. Active in play too: eval shows jumps.
  cfg.events["reset_jump_schedule"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_jump_schedule,
    params={"grid_s": JUMP_SLOT_S, "p_no_jump": JUMP_P_NO_JUMP},
  )

  # Same velocity command, but its play-viewer GUI grows a Jump button that
  # schedules a slot at the next gait gap for the viewed env -- the
  # interactive stand-in for the robot's eventual X button. Shallow field
  # copy, same reasoning as the delayed-action swap in pupper_env_cfg.
  twist = cfg.commands["twist"]
  twist_fields = {f.name: getattr(twist, f.name) for f in dataclasses.fields(twist)}
  cfg.commands["twist"] = mdp.MixedJumpVelocityCommandCfg(
    **twist_fields, jump_slot_s=JUMP_SLOT_S, jump_busy_s=JUMP_REWARD_WINDOW_S
  )
  return cfg
