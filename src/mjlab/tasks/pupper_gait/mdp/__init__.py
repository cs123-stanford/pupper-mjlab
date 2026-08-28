"""Pupper trot-gait task MDP terms."""

from mjlab.tasks.pupper.mdp import *  # noqa: F401, F403

from .gait import (  # noqa: F401
  gait_base_height,
  gait_phase,
  gait_reference_obs,
  gait_tracking,
  pupper_gait_actor_obs,
  pupper_gait_shared_obs,
  reference_offset,
  reference_offset_from_tables,
  reset_gait_shared_obs,
)
from .jump import (  # noqa: F401
  jump_actor_obs,
  jump_airborne_height,
  jump_hold_descent,
  jump_reference_obs,
  jump_reference_offset,
  jump_tracking,
  jump_up_velocity,
  torso_clearance,
)
from .mixed_gaits import (  # noqa: F401
  mixed_gait_actor_obs,
  mixed_gait_reference_obs,
  mixed_gait_tracking,
  mixed_reference_offset,
  roll_asymmetry,
  zero_reference_actor_obs,
)
from .mixed_jump import (  # noqa: F401
  MixedJumpVelocityCommand,
  MixedJumpVelocityCommandCfg,
  air_time_outside_slot,
  base_height_outside_slot,
  foot_slip_outside_slot,
  lin_vel_z_l2_outside_slot,
  mixed_jump_actor_obs,
  mixed_jump_reference_obs,
  mixed_jump_reference_offset,
  mixed_jump_tracking,
  request_jump,
  reset_jump_schedule,
  slot_jump_airborne_height,
  slot_jump_up_velocity,
  stand_still_joint_velocity_outside_slot,
  stand_still_pose_outside_slot,
  track_linear_velocity_windowed,
)
from .warmup import (  # noqa: F401
  hold_base_airborne,
  pretrain_rewards,
)
