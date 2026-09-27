"""Pupper v3 motion tracking (BeyondMimic) environment configuration.

Trains a small policy that makes Pupper follow one reference motion (a trick),
root pose and joints together, with no velocity command. Motions come from
``motion_design.py`` (see ``tricks/sit.py``) and are converted to the
tracking task's npz format by ``Motion.save_npz``.

Differences from the G1 setup:
- Pupper's physics options, 50 Hz control, action scale and latency model, so
  a trick policy targets the same dynamics as the walking policies.
- Body collision: tricks sit, lie down and kneel, so the torso, upper legs and
  lower legs get convex-hull collision shapes (ground contact only) on top of
  the 8 knee and foot balls the walking tasks use.
- No IMU sensors in the Pupper model: base velocities come from the simulator,
  and the actor sees only what the robot can measure (no base position or
  linear velocity), like G1's "No-State-Estimation" variant.
- The robot's latency and hardware spread, as in the walking tasks: a fixed
  16 ms actuator delay, 0-1 control steps of IMU latency on the angular velocity
  and body orientation terms, and randomized PD gains, mass and inertia.
"""

from __future__ import annotations

import dataclasses

import mujoco

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  PUPPER_ACTION_SCALE,
  PUPPER_DECIMATION,
  PUPPER_MUJOCO_OPTIONS,
  get_pupper_robot_cfg,
  get_spec,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.pupper.mdp.latency import (
  PUPPER_ACTION_LATENCY_PHYSICS_STEPS,
  PUPPER_IMU_LATENCY_DIST,
)
from mjlab.tasks.pupper.pupper_env_cfg import _inertial_event, _pd_gain_event
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# Bodies that get a convex-hull collision shape, and the vertex cap per hull.
HULL_BODIES = (
  "base_link",
  "leg_front_r_2", "leg_front_l_2", "leg_back_r_2", "leg_back_l_2",
  "leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3",
)  # fmt: skip
HULL_MAX_VERTICES = 32

# Bodies the tracking reward compares against the reference.
TRACKED_BODIES = (
  "base_link",
  "leg_front_r_2", "leg_front_r_3", "leg_front_l_2", "leg_front_l_3",
  "leg_back_r_2", "leg_back_r_3", "leg_back_l_2", "leg_back_l_3",
)  # fmt: skip
FOOT_BODIES = ("leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3")


def get_spec_with_hulls() -> mujoco.MjSpec:
  """The Pupper model plus convex-hull collision geoms on the torso and legs.

  Each hull reuses its body's visual mesh (MuJoCo collides meshes as their convex
  hull). contype 0 / conaffinity 1 means a hull only touches things that start
  a contact: the ground and the knee/foot balls of other links. Hulls never
  touch each other, so the overlapping torso and upper-leg hulls at the
  shoulders don't lock the robot. density 0 leaves the mass unchanged.
  """
  spec = get_spec()
  meshes = set()
  for body_name in HULL_BODIES:
    body = spec.body(body_name)
    visual = next(g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH)
    body.add_geom(
      name=f"{body_name}_hull",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=visual.meshname,
      pos=visual.pos,
      quat=visual.quat,
      contype=0,
      conaffinity=1,
      condim=3,
      group=3,
      density=0,
      friction=(0.8, 0.02, 0.01),
    )
    meshes.add(visual.meshname)
  for mesh_name in meshes:
    spec.mesh(mesh_name).maxhullvert = HULL_MAX_VERTICES
  return spec


def pupper_tracking_env_cfg(
  motion_file: str = "", play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Pupper flat-ground tracking configuration."""
  cfg = make_tracking_env_cfg()
  cfg.sim = SimulationCfg(
    nconmax=80, njmax=600, mujoco=MujocoCfg(**PUPPER_MUJOCO_OPTIONS)
  )
  cfg.decimation = PUPPER_DECIMATION

  robot = get_pupper_robot_cfg()
  robot.spec_fn = get_spec_with_hulls
  # Fixed actuator delay (see pupper_env_cfg.py). Replace the articulation rather
  # than mutating it: it is a shared module-level object.
  assert robot.articulation is not None
  n = PUPPER_ACTION_LATENCY_PHYSICS_STEPS
  robot.articulation = dataclasses.replace(
    robot.articulation,
    actuators=tuple(
      dataclasses.replace(a, delay_min_lag=n, delay_max_lag=n)
      for a in robot.articulation.actuators
    ),
  )
  cfg.scene.entities = {"robot": robot}
  # Foot-to-foot strikes on the same side are the self-collisions that matter
  # (see pupper_env_cfg.py); penalize them like G1 penalizes any self-contact.
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(
        mode="geom", pattern=r"^leg_front_(r|l)_3_collision$", entity="robot"
      ),
      secondary=ContactMatch(
        mode="geom", pattern=r"^leg_back_(r|l)_3_collision$", entity="robot"
      ),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PUPPER_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = motion_file
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = TRACKED_BODIES
  # Pupper is small: scale G1's reset randomization down.
  motion_cmd.pose_range = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (-0.005, 0.005),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
  }
  motion_cmd.velocity_range = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.1, 0.1),
    "roll": (-0.3, 0.3),
    "pitch": (-0.3, 0.3),
    "yaw": (-0.3, 0.3),
  }
  cfg.events["push_robot"].params["velocity_range"] = motion_cmd.velocity_range

  # Observations: the robot has an IMU (angular velocity, gravity direction) and
  # joint encoders, but no base position or linear velocity.
  actor = cfg.observations["actor"].terms
  actor.pop("motion_anchor_pos_b")
  actor.pop("base_lin_vel")
  # IMU latency: 0 or 1 control steps, equally likely (PUPPER_IMU_LATENCY_DIST).
  imu_lag = len(PUPPER_IMU_LATENCY_DIST) - 1
  actor["base_ang_vel"] = ObservationTermCfg(
    func=envs_mdp.base_ang_vel,
    noise=Unoise(n_min=-0.2, n_max=0.2),
    delay_max_lag=imu_lag,
  )
  actor["motion_anchor_ori_b"] = dataclasses.replace(
    actor["motion_anchor_ori_b"], delay_max_lag=imu_lag
  )
  critic = cfg.observations["critic"].terms
  critic["base_lin_vel"] = ObservationTermCfg(func=envs_mdp.base_lin_vel)
  critic["base_ang_vel"] = ObservationTermCfg(func=envs_mdp.base_ang_vel)
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor, concatenate_terms=True, enable_corruption=True
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^leg_.*_3_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["pd_gains"] = _pd_gain_event()
  cfg.events["inertial"] = _inertial_event()
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.01, 0.01),
  }
  # Pupper is ~0.16 m tall: G1's 0.25 m error thresholds would never trigger.
  cfg.terminations["anchor_pos"].params["threshold"] = 0.08
  cfg.terminations["ee_body_pos"].params["body_names"] = FOOT_BODIES
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.08
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.1
  cfg.rewards["motion_body_pos"].params["std"] = 0.1
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 0.8

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
  return cfg
