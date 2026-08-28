"""Pupper task MDP terms (extends the shared env MDP)."""

from mjlab.envs.mdp import *  # noqa: F401, F403

# Reuse the velocity task's contact/foot reward + command implementations.
from mjlab.tasks.velocity.mdp import (  # noqa: F401
  UniformVelocityCommandCfg,
  commands_vel,
  feet_air_time,
  feet_slip,
  self_collision_cost,
  track_angular_velocity,
  track_linear_velocity,
  variable_posture,
)

from .actions import (  # noqa: F401
  DelayedJointPositionAction,
  DelayedJointPositionActionCfg,
)
from .latency import (  # noqa: F401
  PUPPER_ACTION_LATENCY_DIST,
  PUPPER_ACTION_LATENCY_PHYSICS_STEPS,
  PUPPER_IMU_LATENCY_DIST,
  LatencyBuffer,
)
from .observations import (  # noqa: F401
  PupperObsNoise,
  applied_action,
  get_imu_latency_buffer,
  pupper_actor_obs,
  reset_imu_latency,
)
from .rewards import (  # noqa: F401
  ang_vel_xy_l2,
  base_height,
  forward_speed,
  heading_deviation,
  joint_deviation_l2,
  lin_vel_z_l2,
  pupper_variable_posture,
  stand_still_joint_velocity,
  stand_still_pose,
  track_yaw_velocity,
)
from .velocity_command import (  # noqa: F401
  PupperMaxSpeedCommand,
  PupperMaxSpeedCommandCfg,
  PupperVelocityCommand,
  PupperVelocityCommandCfg,
)
