"""Upload a designed trick motion to Weights & Biases, so training can pull it.

    uv run python -m mjlab.tasks.tracking.config.pupper.upload_motion \\
        src/mjlab/tasks/tracking/config/pupper/tricks/my_trick.npz --name my_trick

The artifact holds motion.npz (for training) and, if it sits next to the npz,
the robot's joint CSV (<stem>_robot.csv -> robot.csv, for plain replay). Then:

    uv run train Mjlab-Tracking-Flat-Pupper-v3 --registry-name <entity>/mjlab/my_trick

and the run uploads the trained trick policy to its Files as policy.json.
"""

from __future__ import annotations

from pathlib import Path

import tyro

import mjlab


def main(
  motion_file: tyro.conf.Positional[Path],
  name: str,
  project: str = "mjlab",
  entity: str | None = None,
) -> None:
  import wandb

  from mjlab.tasks.tracking.config.pupper.motion_design import read_clips

  robot_csv = motion_file.with_name(f"{motion_file.stem}_robot.csv")
  clips = read_clips(motion_file)
  run = wandb.init(
    project=project, entity=entity, job_type="motion-upload", name=f"motion-{name}"
  )
  artifact = wandb.Artifact(
    name,
    type="motion",
    metadata={"clips": [c["name"] for c in clips], "source": str(motion_file)},
  )
  artifact.add_file(str(motion_file), name="motion.npz")
  if robot_csv.exists():
    artifact.add_file(str(robot_csv), name="robot.csv")
  run.log_artifact(artifact)
  artifact.wait()
  path = f"{run.entity}/{run.project}/{name}"
  run.finish()
  print(f"uploaded {path}:{artifact.version}")
  print(f"train it:  uv run train Mjlab-Tracking-Flat-Pupper-v3 --registry-name {path}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
