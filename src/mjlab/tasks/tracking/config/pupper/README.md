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

The easy way to get a motion from where you design it (a laptop, Colab) to
where you train it and on to the robot is Weights & Biases. Upload the motion
once, as an artifact; training pulls it by name:

```bash
uv run python -m mjlab.tasks.tracking.config.pupper.upload_motion \
    src/mjlab/tasks/tracking/config/pupper/tricks/my_trick.npz --name my_trick
uv run train Mjlab-Tracking-Flat-Pupper-v3 --env.scene.num-envs 4096 \
    --registry-name <entity>/mjlab/my_trick
```

The upload includes the robot CSV (`my_trick_robot.csv`, from
`save_robot_csv`) when it sits next to the npz, so the robot can also fetch the
motion for plain replay. A local file works too:

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

## 4. Get it onto the robot

Training uploads the robot's trick policy to the W&B run as `policy.json` at
every checkpoint and on Ctrl+C; the log prints the command to run on the robot,
in your gemini_pupper repo (the Gemini Pupper lab):

```bash
python3 robot/get_trick.py <entity>/mjlab/<run-id> --name my_trick
```

To export a local checkpoint by hand instead:

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

## Clip tricks: several moves, one policy

Some tricks are more than one fixed motion: a pose the robot holds for as long
as you like, with moves it can do from that pose on request. Design each piece
as its own motion and join them with `motion_design.concat_clips`:

```python
library = md.concat_clips(
    [
        md.Clip("get_into_pose", intro, next="hold"),   # plays first
        md.Clip("hold", hold, loop=True),               # repeats until asked
        md.Clip("move_a", move_a, next="hold"),         # returns to the hold
        md.Clip("leave", leave),                        # ends; holds its last frame
    ],
    # training sequence: every clip at least once, with the hold in between
    order=["get_into_pose", "hold", "move_a", "hold", "leave"],
)
library.save_npz("tricks/my_clips.npz")
```

CS 123's lab 6 builds one of these without new keyframes: `tricks/sit_stay_shake.py`
(or the `CS123_Pupper_Sit_Stay_Shake.ipynb` notebook) cuts the worked sit into
sit-down and stand-up clips, holds the sitting pose for the stay, and mirrors
the left paw shake into the right one (swap left and right legs and negate all
three joints: Pupper's left and right motors are mounted mirror-image).

Clips must meet at matching poses (the join refuses a jump of more than
0.05 rad), so the robot can go from the end of any clip to the start of the
next. One policy trains on the whole joined motion; the npz and the exported
JSON carry a clip table. On the robot, the controller plays the first clip,
follows each clip's `next`, loops a looping clip, and switches to a requested
clip at the next clip boundary (topic `/trick_<name>/clip`; progress on
`/trick_<name>/clip_status`). In the Gemini Pupper repo, a trick's `trick.yaml` names
the clips Gemini may request (`actions`) and the one that returns to standing
(`exit_clip`).

## Tips for designing with a coding agent

- Describe the pose in words and numbers ("chest up 35 degrees, hips on the
  ground, front legs straight under the shoulders") and let the agent search
  for keyframe values, as the note at the end of `sit.py` describes.
- Keep the centre of mass over the feet that are on the ground. For a step,
  lean the body away from the lifted foot first.
- Real limits matter: joint limits, motor speed, and keeping the front legs
  under the shoulders when the body tilts. The report and the physics replay
  catch most mistakes.
