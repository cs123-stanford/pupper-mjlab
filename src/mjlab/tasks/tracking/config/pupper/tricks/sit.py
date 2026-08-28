"""Worked example: Pupper sits like a dog, then stands back up.

The sit: hips on the ground, chest up about 35 degrees, weight on straight
front legs. Pupper's hind legs can't fold far enough to sit over feet planted
where it stands, so it first steps each hind foot forward (like a dog shuffling
in), sits, holds, stands, and steps back.

    uv run python -m mjlab.tasks.tracking.config.pupper.tricks.sit

writes, next to this file:
  sit.csv        root pose + joints (BeyondMimic CSV, 50 Hz)
  sit.npz        tracking-task motion file (train with Mjlab-Tracking-Flat-Pupper-v3)
  sit_robot.csv  joint-trajectory trick for the robot's animation player

To design your own trick, copy this file and change the keyframes. Everything is
in metres, degrees and seconds; see motion_design.py for the conventions.
"""

from pathlib import Path

from mjlab.tasks.tracking.config.pupper import motion_design as md

HERE = Path(__file__).resolve().parent

# The sitting pose, found by searching poses where the front feet stay put and
# the butt touches the ground (see the notes at the bottom).
SIT_PITCH = -35.0  # degrees; negative = nose up
BUTT_X = -0.07  # where the butt would touch down before settling (world x, m)
HIND_X = -0.04  # hind feet step forward to here (standing: -0.084)
STEP_HEIGHT = 0.04  # how high a foot lifts while stepping (m)


def keyframes() -> list[md.Keyframe]:
  stand = md.stand_feet()  # where the four feet are when standing
  fr, fl, br, bl = (stand[leg] for leg in md.LEGS)
  h = md.STAND_HEIGHT

  def feet(br_=br, bl_=bl):
    return {"front_r": fr, "front_l": fl, "back_r": br_, "back_l": bl_}

  br_fwd = (HIND_X, br[1], md.FOOT_RADIUS)
  bl_fwd = (HIND_X, bl[1], md.FOOT_RADIUS)
  mid = (br[0] + HIND_X) / 2
  # Start from "butt at BUTT_X on the ground", then let the pose settle so the
  # lowest part (the hind thighs, as on a real dog) just touches the ground.
  sit = md.settle_on_ground(
    md.Keyframe(
      0.0,
      md.root_with_point_on_ground(md.BUTT, (0, SIT_PITCH, 0), x=BUTT_X),
      (0, SIT_PITCH, 0),
      feet=feet(br_fwd, bl_fwd),
    )
  )
  sit_root = sit.root_pos

  kfs = [
    md.stand(0.0),
    md.stand(0.5),
    # Step the right hind foot forward: lean forward-left first so the other
    # three feet carry the weight.
    md.Keyframe(0.8, (0.02, 0.015, h - 0.005), feet=feet()),
    md.Keyframe(
      1.0,
      (0.02, 0.015, h - 0.005),
      feet=feet(br_=(mid, br[1], md.FOOT_RADIUS + STEP_HEIGHT)),
    ),
    md.Keyframe(1.2, (0.02, 0.015, h - 0.005), feet=feet(br_=br_fwd)),
    # Then the left hind foot, leaning forward-right.
    md.Keyframe(1.5, (0.02, -0.015, h - 0.005), feet=feet(br_=br_fwd)),
    md.Keyframe(
      1.7,
      (0.02, -0.015, h - 0.005),
      feet=feet(br_=br_fwd, bl_=(mid, bl[1], md.FOOT_RADIUS + STEP_HEIGHT)),
    ),
    md.Keyframe(1.9, (0.02, -0.015, h - 0.005), feet=feet(br_=br_fwd, bl_=bl_fwd)),
    md.Keyframe(2.2, (0.0, 0.0, h - 0.01), feet=feet(br_fwd, bl_fwd)),
    # Sit: hips back and down, chest up.
    md.Keyframe(3.7, sit_root, (0, SIT_PITCH, 0), feet=feet(br_fwd, bl_fwd)),
    md.Keyframe(5.7, sit_root, (0, SIT_PITCH, 0), feet=feet(br_fwd, bl_fwd)),
    # Stand up again, then step the hind feet back.
    md.Keyframe(7.0, (0.0, 0.0, h - 0.01), feet=feet(br_fwd, bl_fwd)),
    md.Keyframe(7.3, (0.02, 0.015, h - 0.005), feet=feet(br_fwd, bl_fwd)),
    md.Keyframe(
      7.5,
      (0.02, 0.015, h - 0.005),
      feet=feet(br_=(mid, br[1], md.FOOT_RADIUS + STEP_HEIGHT), bl_=bl_fwd),
    ),
    md.Keyframe(7.7, (0.02, 0.015, h - 0.005), feet=feet(br_=br, bl_=bl_fwd)),
    md.Keyframe(8.0, (0.02, -0.015, h - 0.005), feet=feet(br_=br, bl_=bl_fwd)),
    md.Keyframe(
      8.2,
      (0.02, -0.015, h - 0.005),
      feet=feet(br_=br, bl_=(mid, bl[1], md.FOOT_RADIUS + STEP_HEIGHT)),
    ),
    md.Keyframe(8.4, (0.02, -0.015, h - 0.005), feet=feet()),
    md.stand(8.8),
    md.stand(9.2),
  ]
  return kfs


def main() -> None:
  motion = md.build_motion(keyframes(), fps=50.0)
  for key, value in motion.report.items():
    print(f"{key:36} {value}")
  print("wrote", motion.save_csv(HERE / "sit.csv"))
  print("wrote", motion.save_npz(HERE / "sit.npz"))
  print("wrote", motion.save_robot_csv(HERE / "sit_robot.csv"))


if __name__ == "__main__":
  main()

# How the sitting pose was chosen: for pitches from -25 to -50 degrees and butt
# positions along x, solve IK with the front feet fixed at their standing spot
# and the hind feet somewhere on the ground, and keep poses where every foot is
# reached within 2 mm and no collision hull goes below the ground. -35 degrees
# with the butt at x = -0.19 m needs the smallest hind step that still gives a
# clear upright chest. A coding agent can rerun a search like this for a new pose.
