"""Pupper v3 velocity task configuration.

Built on top of mjlab's velocity task so that the "fancy" features (rough-terrain
generator + curriculum, variable-posture pose reward, foot-clearance/height
sensors) remain available for advanced students. The defaults mirror the CS 123
``pupperv3-mjx`` setup: flat ground and a faithful 36-dim proprioceptive actor
observation with 20-step history. Every reward weight ships at zero -- choosing
what to reward, what to penalize, and by how much is the lab (the notebook
passes your weights into training).

The actor observation is intentionally terrain-blind (proprioception only), which
matches the on-robot ``neural_controller`` -- the physical Pupper has no height
scanner. The *critic* keeps mjlab's rich privileged observations, which only
helps training since it is never deployed.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  FOOT_SITE_NAMES,
  PUPPER_ACTION_SCALE,
  PUPPER_DECIMATION,
  PUPPER_MUJOCO_OPTIONS,
  get_pupper_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.pupper import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Foot collision geoms (feet get friction DR) and foot bodies (contact sensor).
_FOOT_GEOMS = tuple(f"{leg}_3_collision" for leg in (
  "leg_front_r", "leg_front_l", "leg_back_r", "leg_back_l"
))  # fmt: skip
_FOOT_BODIES = r"^leg_(front|back)_(r|l)_3$"
_ABDUCTION_JOINTS = (r"leg_.*_2",)  # Mirrors the notebook's [1::3] abduction slice.

# How the robot's command latency is modelled. Both target the same physical
# quantity -- the deploy stack's fixed 15.4 ms bus delay -- and both give a mean
# lag of 16 ms; they differ only in variance.
#
#   "deterministic" - a fixed 4-physics-step delay on the actuator. mjlab
#                     integrates at 4 ms, so the sub-control-step delay is
#                     expressed exactly, with no per-step randomness.
#   "mjx"           - the reference pipeline's ``[0.2, 0.8]`` Bernoulli at
#                     control-step granularity. Reproduces ``pupperv3-mjx`` bit
#                     for bit; keep this to compare against it.
#
# Default is "deterministic". The Bernoulli is a discretization artifact of MJX's
# whole-step buffer rather than jitter the hardware has, and because the realized
# lag is invisible to the policy it cannot learn a phase lead against it -- it
# hedges, which shows up as residual stutter at low authority. See
# :mod:`.mdp.latency`.
PUPPER_LATENCY_MODEL: Literal["deterministic", "mjx"] = "deterministic"


def _pupper_actor_obs_group(add_noise: bool) -> ObservationGroupCfg:
  """The deployable 36-dim proprioceptive actor observation, 20-step history."""
  return ObservationGroupCfg(
    terms={
      "policy": ObservationTermCfg(
        func=mdp.pupper_actor_obs,
        params={
          "command_name": "twist",
          "add_noise": add_noise,
          "imu_latency_distribution": mdp.PUPPER_IMU_LATENCY_DIST,
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,  # Noise is baked into the term (per-subgroup scales).
    history_length=20,
  )


def pupper_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Pupper v3 velocity config on rough terrain (advanced)."""
  cfg = make_velocity_env_cfg()

  # Contacts: Pupper is small, so keep the allocation modest.
  cfg.sim = SimulationCfg(
    nconmax=50,
    njmax=500,
    mujoco=MujocoCfg(**PUPPER_MUJOCO_OPTIONS),
  )
  cfg.decimation = PUPPER_DECIMATION
  cfg.episode_length_s = 10.0  # 500 steps * 0.02 s, matching the notebook.

  ##
  # Robot + scene wiring.
  ##
  cfg.scene.entities = {"robot": get_pupper_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
      )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=_FOOT_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Knee spheres (leg_*_2_collision) vs. ground. Matched by geom name so the
  # foot geoms (children of the knee body) are NOT included. Used to penalize the
  # dog turning by planting its knees on the ground.
  knee_ground_cfg = ContactSensorCfg(
    name="knee_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^leg_(front|back)_(r|l)_2_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  # Foot ball against foot ball. Rare, but on hardware a foot-to-foot strike
  # wrecks the stride, and nothing penalized it -- the model simulates these
  # contacts (every leg geom is contype/conaffinity 1, no exclusion pairs) but no
  # reward saw them.
  #
  # One sensor per side, because ContactSensor resolves ``secondary`` to a *single*
  # element: a pattern matching both back feet silently watches only one of them,
  # and the other pair goes unseen. Sampling reachable poses inside the real joint
  # limits, the same-side front/back pairs are also the only ones that can meet at
  # all -- front_l/back_l closes to -0.030 m and front_r/back_r to -0.024 m, while
  # every left-right pair stays 0.05 m or more apart because abduction runs out of
  # travel first.
  self_collision_cfgs = tuple(
    ContactSensorCfg(
      name=f"self_collision_{side}",
      primary=ContactMatch(
        mode="geom", pattern=rf"^leg_front_{side}_3_collision$", entity="robot"
      ),
      secondary=ContactMatch(
        mode="geom", pattern=rf"^leg_back_{side}_3_collision$", entity="robot"
      ),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    )
    for side in ("r", "l")
  )
  cfg.scene.sensors = (
    (cfg.scene.sensors or ()) + (feet_ground_cfg, knee_ground_cfg) + self_collision_cfgs
  )

  # Turn off sensor debug visualization. The raycast/height sensors otherwise
  # render a dense grid of magenta hit-spheres in the Viser viewer (~160 for the
  # terrain scan), which is purely cosmetic but tanks viewer frame rate. The
  # sensors still compute normally for the critic.
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, (RayCastSensorCfg, TerrainHeightSensorCfg)):
      sensor.debug_vis = False

  cfg.viewer.body_name = "base_link"

  ##
  # Actions: position control, offset by the default pose, scaled by 0.75.
  ##
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PUPPER_ACTION_SCALE

  # Command latency: the applied target lags the policy output, as it does on the
  # bus. Kept in play too -- see the observation note below. Two models, see
  # PUPPER_LATENCY_MODEL.
  if PUPPER_LATENCY_MODEL == "mjx":
    # Shallow field copy: asdict() would deep-convert nested cfg dataclasses.
    fields = {
      f.name: getattr(joint_pos_action, f.name)
      for f in dataclasses.fields(joint_pos_action)
    }
    cfg.actions["joint_pos"] = mdp.DelayedJointPositionActionCfg(
      **fields, latency_distribution=mdp.PUPPER_ACTION_LATENCY_DIST
    )
  else:
    # Same mean lag, expressed exactly instead of as a per-step coin flip. The
    # actuator's own delay is measured in physics steps, so min == max gives a
    # fixed 16 ms. Replace rather than mutate: PUPPER_ACTUATOR is a shared
    # module-level object and mutating it would leak into every other task.
    entity = cfg.scene.entities["robot"]
    assert entity.articulation is not None, "Pupper robot cfg must be articulated"
    n = mdp.PUPPER_ACTION_LATENCY_PHYSICS_STEPS
    # Replace both the actuator *and* the articulation that holds it, rather than
    # assigning through. get_pupper_robot_cfg() returns a fresh EntityCfg but
    # hands it the shared module-level PUPPER_ARTICULATION, so mutating that
    # container leaks the delay into every task built afterwards.
    entity.articulation = dataclasses.replace(
      entity.articulation,
      actuators=tuple(
        dataclasses.replace(a, delay_min_lag=n, delay_max_lag=n)
        for a in entity.articulation.actuators
      ),
    )

  ##
  # Observations: faithful proprioceptive actor, rich privileged critic (kept).
  #
  # Noise and IMU latency stay on in play. The MJX pipeline these tasks port from
  # builds its eval env with the *same* kwargs as training, and both are permanent
  # properties of the robot rather than training-time regularizers. The latency in
  # particular is not optional: its median lag is nonzero, so serving undelayed
  # observations at play time is a materially different plant, not a mild one --
  # which is exactly why a latency-free sim looked healthy while the robot shook.
  ##
  cfg.observations["actor"] = _pupper_actor_obs_group(add_noise=True)

  # The velocity critic reads base velocity from G1's IMU sensors, which the
  # Pupper model doesn't define. Read them from state instead (privileged, and
  # the critic is never deployed).
  critic = cfg.observations["critic"]
  critic.terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel)
  critic.terms["base_ang_vel"] = ObservationTermCfg(func=mdp.base_ang_vel)

  # Under the "mjx" latency model the inherited "actions" term is the policy's
  # *raw* output while the applied motor target is stochastically one step stale,
  # resampled every step per env -- so the two disagree on ~80% of steps and
  # nothing in the observation says which happened, leaving the value function to
  # predict returns from a transition whose input it cannot see. Give it the
  # applied target too; the pair determines the realized lag. Critic-only: the
  # robot cannot know its own bus lag, so the actor must not see this.
  #
  # The deterministic model has no such hidden variable -- the lag is a constant,
  # so the applied target is a fixed function of the policy's own history -- and
  # the term would be redundant. It is also unavailable there: the delay lives on
  # the actuator rather than on a DelayedJointPositionAction.
  if PUPPER_LATENCY_MODEL == "mjx":
    critic.terms["applied_action"] = ObservationTermCfg(func=mdp.applied_action)

  ##
  # Commands: sample yaw-rate directly (no heading controller), Pupper ranges.
  # A share of envs get pure "walk straight" (forward-only) and "turn in place"
  # (yaw-only) commands so the policy sees plenty of each canonical behavior.
  ##
  twist_cmd = mdp.PupperVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    heading_command=False,
    rel_standing_envs=0.02,  # zero_command_probability.
    rel_forward_envs=0.15,  # "walk straight" (positive x, zero y/yaw).
    rel_turn_envs=0.15,  # "turn in place" (zero x/y, |yaw| >= turn_min).
    debug_vis=True,
    ranges=mdp.PupperVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.75, 0.75),
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-2.0, 2.0),
      heading=None,
    ),
  )
  twist_cmd.viz.z_offset = 0.3
  cfg.commands["twist"] = twist_cmd

  ##
  # Domain randomization events.
  ##
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOMS
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.4)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.02, 0.02),
    1: (-0.005, 0.005),
    2: (-0.005, 0.005),
  }
  # Base-velocity kicks, off.
  #
  # pupperv3-mjx does kick (kick_vel 0.2 m/s at 2% per step), but the velocity
  # task's push_robot is a much harsher disturbance: +/-0.5 m/s linear -- 2.5x --
  # plus z and full angular kicks (roll/pitch +/-0.52, yaw +/-0.78 rad/s) that the
  # reference does not apply at all, every 1-3 s. Disabled rather than retuned to
  # 0.2 for now; re-add a linear-only kick if the policy turns out to be fragile
  # to shoves on hardware.
  cfg.events.pop("push_robot", None)

  # PD-gain randomization (kp x[0.6,1.1], kd x[0.8,1.5]).
  cfg.events["pd_gains"] = _pd_gain_event()
  # Mass + inertia randomization (density scale x[0.9,1.3], consistent inertia).
  cfg.events["inertial"] = _inertial_event()
  # Clear the IMU latency history so a fresh episode cannot read a stale frame
  # from the previous one. The action latency buffer is reset by the action term.
  cfg.events["reset_imu_latency"] = EventTermCfg(
    mode="reset", func=mdp.reset_imu_latency, params={}
  )

  ##
  # Rewards: the Pupper term vocabulary, every weight at zero.
  ##
  _set_pupper_rewards(cfg)

  ##
  # Terminations: fall over (30 deg) is the notebook's terminal_body_angle.
  ##
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(30.0)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}

  return cfg


def pupper_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Pupper v3 velocity config on flat ground (default, matches the notebook)."""
  cfg = pupper_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No terrain to scan: drop the raycaster and its critic observation.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["critic"].terms.pop("height_scan", None)

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  return cfg


def _pd_gain_event() -> EventTermCfg:
  return EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
      "kp_range": (0.6, 1.1),
      "kd_range": (0.8, 1.5),
      "operation": "scale",
    },
  )


def _inertial_event() -> EventTermCfg:
  # alpha scales mass & inertia by exp(2*alpha); [ln0.9, ln1.3]/2 -> x[0.9,1.3].
  return EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (-0.0527, 0.1312),
    },
  )


def _set_pupper_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Define the Pupper reward vocabulary -- every weight ships at zero.

  Terms already present from the velocity task are re-parameterized for the
  Pupper; missing terms are added. What each term *measures* is given here and
  in the term functions; which terms matter, their signs, and their magnitudes
  are yours to choose (the lab notebook passes your weights into training).
  """
  r = cfg.rewards

  # --- Task tracking (present in velocity task). ---
  # Reward = exp(-||cmd - vel||^2 / std^2); smaller std = stricter tracking.
  r["track_linear_velocity"].weight = 0.0
  r["track_linear_velocity"].params["std"] = math.sqrt(0.1)

  # Yaw tracking uses the reference implementation's form instead of the shared
  # velocity term, which folds the squared roll/pitch rate into the yaw error and
  # left the term dominated by trot body wobble rather than by yaw (see
  # mdp.track_yaw_velocity). Renamed rather than re-parameterized in place: it
  # measures a different quantity now, and logging it under the old name would make
  # curves from before and after look comparable when they are not.
  del r["track_angular_velocity"]
  r["track_yaw_velocity"] = RewardTermCfg(
    func=mdp.track_yaw_velocity,
    weight=0.0,
    params={"command_name": "twist", "sigma": 0.25},
  )
  r["upright"].weight = 0.0  # ~ tracking_orientation (upright target).
  r["upright"].params["asset_cfg"].body_names = ("base_link",)

  # --- Regularization (present). ---
  r["action_rate_l2"].weight = 0.0
  # Joint-limit proximity: dropped rather than blanked -- the joint-limit
  # termination already covers it.
  del r["dof_pos_limits"]

  # --- Foot behavior (present; need the contact sensor + foot sites). ---
  r["air_time"].weight = 0.0
  r["air_time"].params["command_threshold"] = 0.05
  r["foot_slip"].weight = 0.0
  r["foot_slip"].params["asset_cfg"].site_names = FOOT_SITE_NAMES

  # --- Standing-posture reward (variable_posture). ---
  # reward = exp(-mean(error^2/std^2)); larger std = weaker posture hold.
  # Pose tracking fights the leg motion locomotion needs, so hold posture firmly
  # only when actually standing still, keep just a light hold while walking, and
  # effectively turn it OFF while running (huge std -> reward ~1 regardless of
  # pose).
  if "pose" in r:
    r["pose"].func = mdp.pupper_variable_posture
    r["pose"].weight = 0.0
    r["pose"].params["asset_cfg"].joint_names = (".*",)
    r["pose"].params["std_standing"] = {".*": 0.1}
    r["pose"].params["std_walking"] = {r".*_1": 7, r".*_2": 0.5, r".*_3": 7}
    r["pose"].params["std_running"] = {r".*_1": 10, r".*_2": 0.5, r".*_3": 10}
    # Turn-in-place regime (linear < 0.1, |yaw| > 0.5): free the abduction (_1)
    # and knee (_3) joints the turn needs, hold the hip (_2) moderately.
    r["pose"].params["std_turning"] = {r".*_1": 0.5, r".*_2": 0.5, r".*_3": 0.5}
    r["pose"].params["turn_linear_max"] = 0.1
    r["pose"].params["turn_angular_min"] = 0.1

  # --- Extra velocity-task terms, dropped to keep the reward menu focused. ---
  # Re-add any of them from the upstream velocity task if your design wants one.
  for name in ("foot_clearance", "foot_swing_height", "soft_landing",
               "body_ang_vel", "angular_momentum"):  # fmt: skip
    r.pop(name, None)

  # --- Added Pupper-specific terms. ---
  r["orientation_l2"] = RewardTermCfg(func=mdp.flat_orientation_l2, weight=0.0)
  r["lin_vel_z_l2"] = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=0.0)
  r["ang_vel_xy_l2"] = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=0.0)
  r["joint_torques_l2"] = RewardTermCfg(func=mdp.joint_torques_l2, weight=0.0)
  r["joint_acc_l2"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=0.0)
  r["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=0.0)
  r["stand_still_pose"] = RewardTermCfg(
    func=mdp.stand_still_pose,
    weight=0.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  r["stand_still_joint_velocity"] = RewardTermCfg(
    func=mdp.stand_still_joint_velocity,
    weight=0.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  r["abduction_angle"] = RewardTermCfg(
    func=mdp.joint_deviation_l2,
    weight=0.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ABDUCTION_JOINTS)},
  )
  # Knee spheres planted on the ground, and legs crossed into each other:
  # counts sim substeps where the geoms touch. Both risk the hardware.
  for _side in ("r", "l"):
    r[f"self_collision_{_side}"] = RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=0.0,
      params={"sensor_name": f"self_collision_{_side}", "force_threshold": 1.0},
    )
  r["knee_ground_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=0.0,
    params={"sensor_name": "knee_ground_contact", "force_threshold": 1.0},
  )
  # Torso height against its nominal standing height (~0.145 m, measured).
  r["base_height"] = RewardTermCfg(
    func=mdp.base_height,
    weight=0.0,
    params={"target_height": 0.145, "std": 0.05},
  )
