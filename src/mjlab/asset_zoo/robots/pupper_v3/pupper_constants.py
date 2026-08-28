"""Pupper v3 constants.

Ported from the CS 123 ``pupperv3-mjx`` training stack. The physics knobs mirror
the ``pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations`` MJX
model that the original Colab notebook trained against, so a policy trained here
targets the same dynamics the on-robot ``neural_controller`` deploys against.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PUPPER_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "pupper_v3" / "xmls" / "pupper_v3.xml"
)
assert PUPPER_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(PUPPER_XML))


##
# Joint ordering.
#
# This order matches the on-robot ``neural_controller`` ``joint_names`` list and
# the MJX training env, and therefore the observation/action layout. Do not
# reorder without updating the exporter and deploy config in lockstep.
##

JOINT_NAMES: tuple[str, ...] = (
  "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
  "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
  "leg_back_r_1", "leg_back_r_2", "leg_back_r_3",
  "leg_back_l_1", "leg_back_l_2", "leg_back_l_3",
)  # fmt: skip

# Default standing pose (abduction, hip, knee) per leg, in JOINT_NAMES order.
DEFAULT_POSE: tuple[float, ...] = (
  0.26, 0.0, -0.52,
  -0.26, 0.0, 0.52,
  0.26, 0.0, -0.52,
  -0.26, 0.0, 0.52,
)  # fmt: skip

FOOT_SITE_NAMES: tuple[str, ...] = (
  "leg_front_r_3_foot_site",
  "leg_front_l_3_foot_site",
  "leg_back_r_3_foot_site",
  "leg_back_l_3_foot_site",
)

##
# Position-servo + passive joint properties.
#
# The MJX model used a MuJoCo built-in position actuator with gainprm=[kp,0,0],
# biasprm=[0,-kp,-kd], forcerange=[-3,3], on top of passive joint damping,
# frictionloss and armature from the model defaults. kp/kd here are the notebook's
# training overrides (position_control_kp=5.5, dof_damping=0.25).
##

PUPPER_KP = 5.5
PUPPER_KD = 0.25
PUPPER_EFFORT_LIMIT = 3.0  # Nm, from forcerange="-3 3".

PUPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=JOINT_NAMES,
  stiffness=PUPPER_KP,
  damping=PUPPER_KD,
  effort_limit=PUPPER_EFFORT_LIMIT,
  armature=0.0016,
  frictionloss=0.125,
  viscous_damping=0.01,
)

##
# Action scale.
#
# On-robot control law: motor_target = default_pose + action_scale * tanh_out,
# clipped to joint limits. Matches policy_config.action_scale in the notebook.
##

PUPPER_ACTION_SCALE: float = 0.75

# Per-frame observation width for the deployable actor policy:
# ang_vel(3) + proj_grav(3) + command(3) + desired_world_z(3) + q-default(12)
# + last_action(12). History of these frames forms the full policy input.
SINGLE_OBS_DIM: int = 36

##
# Collision config.
#
# The source model has self-collision enabled with condim=3 spheres at each knee
# (``*_2_collision``) and foot (``*_3_collision``). Feet get priority so their
# friction wins over the ground. All other geoms (visual meshes) are disabled.
##

PUPPER_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority={r".*_3_collision$": 1},
  friction={
    r".*_3_collision$": (0.8, 0.02, 0.01),
    r".*_2_collision$": (0.8, 0.02, 0.01),
  },
  solimp=(0.015, 1.0, 0.031),
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.18),
  joint_pos={
    name: angle for name, angle in zip(JOINT_NAMES, DEFAULT_POSE, strict=True)
  },
  joint_vel={".*": 0.0},
)

##
# Recommended simulation options.
#
# These reproduce the MJX ``<option>`` block the notebook trained with:
#   <option cone="pyramidal" impratio="10" iterations="2" ls_iterations="5"
#           timestep="0.004"> <flag eulerdamp="disable"/> </option>
# Consume via ``MujocoCfg(**PUPPER_MUJOCO_OPTIONS)`` in the env config. Getting
# these wrong (e.g. mjlab's default well-solved contacts) changes the dynamics
# the policy sees and degrades sim2real transfer.
##

PUPPER_MUJOCO_OPTIONS: dict = {
  "timestep": 0.004,
  "integrator": "euler",
  "cone": "pyramidal",
  "impratio": 10.0,
  "iterations": 2,
  "ls_iterations": 5,
  "disableflags": ("eulerdamp",),
}

# Control runs at 50 Hz: 0.02 s / 0.004 s = 5 physics steps per control step.
PUPPER_DECIMATION: int = 5

##
# Final config.
##

PUPPER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(PUPPER_ACTUATOR,),
  soft_joint_pos_limit_factor=1.0,
)


def get_pupper_robot_cfg() -> EntityCfg:
  """Get a fresh Pupper v3 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(PUPPER_COLLISION,),
    spec_fn=get_spec,
    articulation=PUPPER_ARTICULATION,
  )


# Per-joint action scale map (uniform across joints), for parity with mjlab's
# JointPositionActionCfg scale API.
PUPPER_ACTION_SCALE_MAP: dict[str, float] = {
  name: PUPPER_ACTION_SCALE for name in JOINT_NAMES
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pupper_robot_cfg())
  viewer.launch(robot.spec.compile())
