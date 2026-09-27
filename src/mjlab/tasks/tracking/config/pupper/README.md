# Pupper tricks: design, replay, train

A trick is a **reference motion**: the body's position and orientation plus the
12 joint angles over time. One motion file serves three uses:

| Use | What it needs | Tool |
| --- | --- | --- |
| Replay on the robot | Joint angles only | `Motion.save_robot_csv` |
| Replay in simulation, with physics | Root pose + joints | `play_motion.py` |
| Train a tracking policy (BeyondMimic) | Root pose + joints + body states | `Mjlab-Tracking-Flat-Pupper-v3` |

Plain replay drives the joints to the reference open loop. It works for gentle
motions. When it wobbles or falls, train a small policy that tracks the motion
while keeping balance; that is how physically awkward motions become feasible.

## 1. Design the motion

Write keyframes: where the body is, and where each foot touches the ground.
Inverse kinematics fills in the joint angles, so planted feet stay planted.
Start from the worked example, a dog sit:

```bash
cp src/mjlab/tasks/tracking/config/pupper/tricks/sit.py \
   src/mjlab/tasks/tracking/config/pupper/tricks/my_trick.py
uv run python -m mjlab.tasks.tracking.config.pupper.tricks.my_trick
```

Conventions (see `motion_design.py`): metres, degrees, seconds; x forward, y
left, z up; the ground is z = 0; negative pitch lifts the nose. Useful helpers:

- `md.stand(t)`: the default standing pose.
- `md.stand_feet()`: where the four feet are when standing.
- `md.root_with_point_on_ground(md.BUTT, rpy, x=...)`: a root that puts a body
  point on the ground.
- `md.settle_on_ground(keyframe)`: lower or raise a pose until its lowest part
  (body or legs) just touches the ground, with the feet re-solved.
- A leg left out of `feet` is free: give its joint angles in `joints` to raise a
  paw.

Every frame is kept above the ground automatically. Read the printed report
before moving on:

| Report line | Good value |
| --- | --- |
| `max_ik_error_m` | under 0.002: every foot reaches its target |
| `lowest_foot_bottom_m`, `lowest_body_point_m` | 0 or above |
| `max_joint_speed_rad_s` | under about 15; the motors top out near 20 |
| `fraction_of_joint_samples_at_limit` | 0 |

The script writes `my_trick.csv` (root + joints), `my_trick.npz` (for training)
and `my_trick_robot.csv` (for the robot).

## 2. Replay it with physics

```bash
uv run python -m mjlab.tasks.tracking.config.pupper.play_motion \
    src/mjlab/tasks/tracking/config/pupper/tricks/my_trick.npz
```

It starts the robot at the motion's first frame, drives the joints to the
reference with the robot's trick-player gains, and prints the error over time
and a verdict:

- **HOLDS UP**: plain replay follows the reference; the robot CSV is enough.
- **WOBBLES**: it stays up but drifts; replay may be fine, a policy will be tighter.
- **FALLS**: plain replay won't work; make the motion gentler, or train it.

Watch it with `--viewer viser` (add `--share` on Colab): the robot plus a
translucent ghost of the reference. `--no-physics` shows the reference alone.

## 3. Train it (when replay isn't enough)

```bash
uv run train Mjlab-Tracking-Flat-Pupper-v3 --env.scene.num-envs 4096 \
    --env.commands.motion.motion-file src/mjlab/tasks/tracking/config/pupper/tricks/my_trick.npz
uv run play Mjlab-Tracking-Flat-Pupper-v3 --wandb-run-path <entity>/mjlab/<run-id> \
    --env.commands.motion.motion-file src/mjlab/tasks/tracking/config/pupper/tricks/my_trick.npz
```

Score a trained checkpoint with the same error timeline and verdict as plain
replay, to see how much training bought you:

```bash
uv run python -m mjlab.tasks.tracking.config.pupper.play_motion \
    src/mjlab/tasks/tracking/config/pupper/tricks/my_trick.npz \
    --checkpoint logs/rsl_rl/pupper_tracking/<run>/model_3000.pt
```

For the sit, plain replay wobbles (worst 7 cm, 24 degrees off the reference)
while a policy trained for 500 iterations holds within 2 cm and 4 degrees.

The policy sees the reference's joint angles and velocities, how its body
orientation differs from the reference's, and the same proprioception the robot
has (angular velocity, joint angles and velocities, last action); no velocity
command and no base position. It is rewarded for matching the body and joint
motion. Physics matches the walking tasks, with one addition: the torso, upper
legs and lower legs have convex-hull collision shapes (ground contact only), so
tricks can sit, kneel and lie down.

## 4. Export it for the robot

```bash
uv run python -m mjlab.tasks.tracking.config.pupper.export_trick \
    --checkpoint logs/rsl_rl/pupper_tracking/<run>/model_3000.pt \
    --motion-file src/mjlab/tasks/tracking/config/pupper/tricks/my_trick.npz \
    --name my_trick --out my_trick_policy.json
```

The JSON carries the network, its observation layout and the reference motion
itself; the robot's controller (pupper_gait_deploy) plays the motion while the
policy runs. The exporter checks that the exported network reproduces the
trained one before writing the file.

## Tips for designing with a coding agent

- Describe the pose in words and numbers ("chest up 35 degrees, hips on the
  ground, front legs straight under the shoulders") and let the agent search
  for keyframe values, as the note at the end of `sit.py` describes.
- Keep the centre of mass over the feet that are on the ground. For a step,
  lean the body away from the lifted foot first.
- Real limits matter: joint limits, motor speed, and keeping the front legs
  under the shoulders when the body tilts. The report and the physics replay
  catch most mistakes.
