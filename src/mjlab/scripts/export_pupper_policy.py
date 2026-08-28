"""Export a trained Pupper policy to the on-robot ``neural_controller`` JSON.

Handles both Pupper task families:

* the velocity tasks, whose deploy frame is the 36-dim proprio frame, and
* the gait tasks, whose frame is 48-dim (proprio + the 12-dim reference offset the
  policy tracks). Those also get a ``gait_reference`` block so the robot can
  reproduce the reference itself -- see :mod:`mjlab.tasks.pupper_gait.export`.

The frame layout is detected from the env's actor observation term, so the right
contract is emitted without a flag. A parity check against the source actor runs on
real observations before the file is written.

Example::

  uv run export-pupper-policy Mjlab-Trot-Bumpy-Pupper-v3 --wandb-run-path mjlab/pdfzwf3l --output policy.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.pupper.export import (
  command_clip_from_env,
  export_pupper_policy_from_env,
  json_forward,
  to_robot_order,
)
from mjlab.tasks.pupper_gait.export import (
  GAIT_OBS_COMPONENTS,
  GAIT_SINGLE_OBS_DIM,
  gait_frame_kind,
  gait_params_from_env,
  gait_reference_metadata,
  heading_hold_metadata,
  jump_reference_metadata,
  jump_reference_offset_numpy,
  jump_slot_metadata,
  mixed_jump_reference_offset_numpy,
  reference_offset_numpy,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportConfig:
  wandb_run_path: str | None = None
  """W&B run to pull the checkpoint from, e.g. ``mjlab/pdfzwf3l``."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the run (e.g. ``model_4000.pt``)."""
  checkpoint_file: str | None = None
  """Local checkpoint path, used instead of the W&B run."""
  output: str = "policy.json"
  device: str | None = None
  use_imu: bool = True
  log_root: str = "logs/rsl_rl"
  parity_tol: float = 1e-3
  """Max allowed abs error between the exported JSON and the source actor."""
  upload_wandb: bool = False
  """Upload the exported JSON to the W&B run's files, so the robot can fetch it.

  The robot cannot run this exporter (it needs a GPU and MuJoCo-Warp), so the run
  is the handoff point: see the ``pupper_gait_deploy`` repo's ``download_policy.py``.
  """
  golden_output: str | None = None
  """Also write the deploy parity fixture (gait tasks only).

  This is the ``gait_golden.json`` the robot-side C++ test checks itself against;
  regenerate it whenever the reference tables or gait parameters change.
  """
  clip_vx: tuple[float, float] | None = None
  """Override the deploy command clip on vx (default: the trained range).

  The robot clamps the teleop command to this before the observation and the
  gait reference see it. Tightening it caps the deployed policy below its
  trained band -- e.g. ``--clip-vx "(-0.49, 0.49)"`` ships a mixed-gaits
  policy that only ever trots.
  """
  clip_vy: tuple[float, float] | None = None
  """Override the deploy command clip on vy (default: the trained range)."""
  clip_wz: tuple[float, float] | None = None
  """Override the deploy command clip on yaw rate (default: the trained range)."""


def _is_gait_env(env: ManagerBasedRlEnv) -> bool:
  """True if the actor frame carries the gait reference offset.

  Covers all 48-dim families: the trot/gallop tasks, the mixed-reference ones
  (MixedGaits, StableGait), and the one-shot jump task -- see
  ``gait_frame_kind``.
  """
  return gait_frame_kind(env) is not None


# Cases the deploy fixture covers: standing, forward/backward trot, turn-driven
# reversal, gallop in both directions, plus a random sweep. These are the branches
# of the reference computation the robot has to reproduce.
_GOLDEN_CASES: tuple[tuple[float, float, float], ...] = (
  (0.0, 0.0, 0.0),  # standing: zero blend.
  (0.3, 0.0, 0.0),  # forward trot.
  (-0.3, 0.0, 0.0),  # backward trot: time-reversed phase.
  (0.02, 0.0, 0.9),  # turn left, |vx| under the direction threshold.
  (0.02, 0.0, -0.9),  # turn right: reversal driven by the yaw sign.
  (0.6, 0.0, 0.0),  # gallop.
  (-0.9, 0.1, 0.2),  # backward gallop.
  (0.001, 0.0, 0.0),  # below blend_speed: partial blend.
)
_GOLDEN_N_RANDOM = 40
_GOLDEN_TOL = 1e-4

# Phase-clock span to sample, in policy steps. This is the *training* episode
# length (10 s at 50 Hz), not ``env.max_episode_length``: play configs set the
# episode to ~forever, and the training term keeps its clock in float32, which
# quantizes the phase badly once t grows large. The robot's C++ uses double, so a
# long-uptime phase is more accurate than sim's, not less -- but the fixture can
# only pin down agreement where sim is still exact.
_GOLDEN_MAX_STEPS = 500


def make_gait_golden(env: ManagerBasedRlEnv, gait: dict) -> dict:
  """Build the C++ parity fixture, cross-checked against the live env's own term.

  ``expected`` holds the deploy-path (float64) values, which is what the robot's C++
  should reproduce essentially exactly. Each one is first checked against the
  training-time term computed by the env itself; the residual there is float32 sim
  noise, and is recorded in the fixture so it is auditable rather than folded away.
  """
  from mjlab.tasks.pupper_gait.mdp.gait import reference_offset

  rng = np.random.default_rng(0)
  commands = np.concatenate(
    [
      np.asarray(_GOLDEN_CASES, dtype=np.float64),
      rng.uniform([-1.0, -0.5, -2.0], [1.0, 0.5, 2.0], size=(_GOLDEN_N_RANDOM, 3)),
    ]
  )
  n = len(commands)
  if env.num_envs < n:
    raise ValueError(
      f"Need at least {n} envs to build the fixture, got {env.num_envs}."
    )
  steps = rng.integers(0, _GOLDEN_MAX_STEPS, size=n)

  params = gait_params_from_env(env)
  env.episode_length_buf[:n] = torch.as_tensor(steps, device=env.device)
  twist = env.command_manager.get_term("twist")
  assert isinstance(twist, UniformVelocityCommand)
  # The fixture maps raw command triples to offsets, so bypass the heading hold
  # -- otherwise get_command() would return the (stale) corrected buffer rather
  # than the commands set below. Deploy runs its own identical hold *upstream*
  # of this computation, so the raw mapping is the right thing to pin down.
  twist.cfg.heading_hold_kp = None
  twist.vel_command_b[:n, :3] = torch.as_tensor(
    commands, dtype=torch.float32, device=env.device
  )

  if gait_frame_kind(env) == "mixed_jump":
    # The composite equals the mixed reference with no slot scheduled; clear
    # the reset-sampled schedule so the gait cases pin exactly that.
    import mjlab.tasks.pupper_gait.mdp.mixed_jump as mixed_jump_mod

    mixed_jump_mod._slot_starts(env)[:] = float("inf")

  if gait_frame_kind(env) in ("mixed", "mixed_jump"):
    from mjlab.tasks.pupper_gait.mdp.mixed_gaits import mixed_reference_offset

    train = (
      mixed_reference_offset(
        env,
        "twist",
        params["frequency"],
        params["blend_speed"],
        int(params["n_samples"]),
        params["gallop_speed"],
      )[:n]
      .cpu()
      .numpy()
    )
  else:
    train = (
      reference_offset(
        env,
        "twist",
        params["frequency"],
        params["blend_speed"],
        int(params["n_samples"]),
        params["gallop_speed"],
        params["gallop_freq_mult"],
      )[:n]
      .cpu()
      .numpy()
    )

  default = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
  t = steps * env.step_dt
  deploy = reference_offset_numpy(
    gait, default, t, commands[:, 0], commands[:, 1], commands[:, 2]
  )

  residual = float(np.abs(deploy - train).max())
  print(f"[INFO]: deploy-vs-training reference residual = {residual:.2e}")
  if residual >= _GOLDEN_TOL:
    raise RuntimeError(
      f"Deploy reference disagrees with the training term ({residual:.2e}); "
      "the robot would track a different gait."
    )

  return {
    "_comment": (
      "Deploy parity fixture generated by mjlab's export-pupper-policy "
      "--golden-output. Regenerate rather than hand-editing."
    ),
    "max_residual_vs_training": residual,
    "gait_reference": gait,
    "default_joint_pos": default.tolist(),
    "cases": [
      {
        "t": float(t[i]),
        "vx": float(commands[i, 0]),
        "vy": float(commands[i, 1]),
        "yaw": float(commands[i, 2]),
        "expected": deploy[i].tolist(),
      }
      for i in range(n)
    ],
  }


# Trigger-clock instants the jump fixture pins down, in policy steps: the idle
# crouch, the hold edge, the launch sweep, flight, and the end-of-episode hold.
_JUMP_GOLDEN_STEPS = (0, 5, 14, 15, 16, 20, 25, 30, 40, 59)
# Deploy-only long-uptime cases (seconds since trigger): the landing hold, far
# past anything training can produce inside a 1.2 s episode. The C++ must pin
# the pose one full wrap later -- the same crouch -- forever.
_JUMP_GOLDEN_EXTRA_T = (5.0, 600.0)
_JUMP_GOLDEN_N_RANDOM = 30


def make_jump_golden(env: ManagerBasedRlEnv, jump: dict) -> dict:
  """Build the C++ parity fixture for the one-shot jump reference.

  Same contract as :func:`make_gait_golden`: ``expected`` is the deploy-path
  (float64) computation, each in-episode case cross-checked against the
  training-time term evaluated by the live env. Cases carry only ``t`` -- the
  jump reference has no command coupling.
  """
  from mjlab.tasks.pupper_gait.mdp.jump import jump_reference_offset

  rng = np.random.default_rng(0)
  steps = np.concatenate(
    [
      np.asarray(_JUMP_GOLDEN_STEPS, dtype=np.int64),
      rng.integers(0, 60, size=_JUMP_GOLDEN_N_RANDOM),
    ]
  )
  n = len(steps)
  if env.num_envs < n:
    raise ValueError(
      f"Need at least {n} envs to build the fixture, got {env.num_envs}."
    )

  params = gait_params_from_env(env)
  env.episode_length_buf[:n] = torch.as_tensor(steps, device=env.device)
  train = (
    jump_reference_offset(
      env,
      params["frequency"],
      int(params["n_samples"]),
      params["crouch_hold_s"],
    )[:n]
    .cpu()
    .numpy()
  )

  default = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
  t = steps * env.step_dt
  deploy = jump_reference_offset_numpy(jump, default, t)

  residual = float(np.abs(deploy - train).max())
  print(f"[INFO]: deploy-vs-training jump reference residual = {residual:.2e}")
  if residual >= _GOLDEN_TOL:
    raise RuntimeError(
      f"Deploy jump reference disagrees with the training term ({residual:.2e}); "
      "the robot would play a different jump."
    )

  t_all = np.concatenate([t, np.asarray(_JUMP_GOLDEN_EXTRA_T, dtype=np.float64)])
  expected_all = jump_reference_offset_numpy(jump, default, t_all)
  return {
    "_comment": (
      "Deploy parity fixture generated by mjlab's export-pupper-policy "
      "--golden-output. Regenerate rather than hand-editing."
    ),
    "max_residual_vs_training": residual,
    "jump_reference": jump,
    "default_joint_pos": default.tolist(),
    "cases": [
      {"t": float(t_all[i]), "expected": expected_all[i].tolist()}
      for i in range(len(t_all))
    ],
  }


# Slot instants the composite fixture pins, as (t_in, vx): pre-window, the
# entry fade, launch, flight, the exit fade, just-after-exit, and long-after,
# across walking / fast / turning / stationary commands.
_SLOT_CASE_T_IN = (-0.02, 0.02, 0.3, 0.5, 0.76, 0.82, 1.2)
_SLOT_CASE_CMDS = ((0.4, 0.0, 0.0), (1.2, 0.0, 0.0), (0.02, 0.0, 0.9), (0.0, 0.0, 0.0))


def make_slot_golden(env: ManagerBasedRlEnv, gait: dict, jump_slot: dict) -> list:
  """Jump-slot composite cases, cross-checked against the training term."""
  import mjlab.tasks.pupper_gait.mdp.mixed_jump as mixed_jump_mod
  from mjlab.tasks.pupper_gait.mdp.mixed_jump import mixed_jump_reference_offset

  slot_start = 1.5
  cases = []
  params = gait_params_from_env(env)
  twist = env.command_manager.get_term("twist")
  assert isinstance(twist, UniformVelocityCommand)
  twist.cfg.heading_hold_kp = None
  default = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()

  combos = [(t_in, cmd) for t_in in _SLOT_CASE_T_IN for cmd in _SLOT_CASE_CMDS]
  n = len(combos)
  assert env.num_envs >= n
  buf = mixed_jump_mod._slot_starts(env)
  buf[:] = float("inf")
  buf[:n, 0] = slot_start
  steps = torch.tensor(
    [round((slot_start + t_in) / env.step_dt) for t_in, _ in combos],
    device=env.device,
  )
  env.episode_length_buf[:n] = steps
  cmds = np.array([c for _, c in combos], dtype=np.float64)
  twist.vel_command_b[:n, :3] = torch.as_tensor(
    cmds, dtype=torch.float32, device=env.device
  )

  train = (
    mixed_jump_reference_offset(
      env,
      "twist",
      params["frequency"],
      params["blend_speed"],
      int(params["n_samples"]),
      params["gallop_speed"],
      float(jump_slot["playback_s"]),
      float(jump_slot["active_s"]),
      float(jump_slot["cross_fade_s"]),
    )[:n]
    .cpu()
    .numpy()
  )
  t = steps.cpu().numpy() * env.step_dt
  deploy = mixed_jump_reference_offset_numpy(
    gait,
    jump_slot,
    default,
    t,
    cmds[:, 0],
    cmds[:, 1],
    cmds[:, 2],
    np.full(n, slot_start),
  )
  residual = float(np.abs(deploy - train).max())
  print(f"[INFO]: deploy-vs-training slot composite residual = {residual:.2e}")
  if residual >= _GOLDEN_TOL:
    raise RuntimeError(
      f"Slot composite disagrees with the training term ({residual:.2e})."
    )
  buf[:] = float("inf")
  for i in range(n):
    cases.append(
      {
        "t": float(t[i]),
        "vx": float(cmds[i, 0]),
        "vy": float(cmds[i, 1]),
        "yaw": float(cmds[i, 2]),
        "slot_start": slot_start,
        "expected": deploy[i].tolist(),
      }
    )
  return cases


def run_export(task_id: str, cfg: ExportConfig) -> Path:
  # IEEE FP32, not TF32: the parity check compares a float64 re-implementation of
  # the robot's network against this process's torch forward, and TF32 matmuls
  # (10-bit mantissa) put ~1e-3 of noise on the torch side -- enough to swamp the
  # thing we are actually trying to detect. Export is not performance-sensitive.
  configure_torch_backends(allow_tf32=False)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  # One env is enough to export; the parity fixture evaluates one case per env.
  env_cfg.scene.num_envs = (
    len(_GOLDEN_CASES) + _GOLDEN_N_RANDOM if cfg.golden_output is not None else 1
  )

  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
  else:
    if cfg.wandb_run_path is None:
      raise ValueError(
        "`wandb_run_path` is required when `checkpoint_file` is not provided."
      )
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    resume_path, _ = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
  print(f"[INFO]: Loading checkpoint: {resume_path}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  actor = runner.get_inference_policy(device=device).eval()

  kind = gait_frame_kind(env)
  is_gait = kind is not None
  single_obs_dim = GAIT_SINGLE_OBS_DIM if is_gait else None

  command_clip = command_clip_from_env(env)
  overrides = {"vx": cfg.clip_vx, "vy": cfg.clip_vy, "wz": cfg.clip_wz}
  if any(v is not None for v in overrides.values()):
    command_clip = dict(command_clip or {})
    command_clip.update({k: v for k, v in overrides.items() if v is not None})

  policy = export_pupper_policy_from_env(
    actor,
    env,
    use_imu=cfg.use_imu,
    single_obs_dim=single_obs_dim,
    obs_components=GAIT_OBS_COMPONENTS if is_gait else None,
    command_clip=command_clip,
  )
  if command_clip is not None:
    print(f"[INFO]: Stamping command_clip: {command_clip}")
  if kind == "jump":
    policy["jump_reference"] = jump_reference_metadata(env)
    print(
      "[INFO]: Jump task detected: emitting a 48-dim frame with the "
      f"{policy['jump_reference']['n_samples']}-sample one-shot jump table "
      f"(trigger: joy button {policy['jump_reference']['trigger_button']})."
    )
  elif kind == "mixed_jump":
    policy["gait_reference"] = gait_reference_metadata(env)
    policy["jump_slot"] = jump_slot_metadata(env)
    print(
      "[INFO]: MixedGaitsJump detected: gait_reference (trot/fast/lift) plus a "
      f"jump_slot block (grid {policy['jump_slot']['grid_s']:.2f}s, window "
      f"{policy['jump_slot']['active_s']:.2f}s, trigger button "
      f"{policy['jump_slot']['trigger_button']})."
    )
  elif is_gait:
    policy["gait_reference"] = gait_reference_metadata(env)
    if "gallop_back_table" in policy["gait_reference"]:
      tables = "trot/fast-fwd/fast-back/lift"
    elif "lift_table" in policy["gait_reference"]:
      tables = "trot/fast/lift"
    else:
      tables = "trot/gallop"
    print(
      "[INFO]: Gait task detected: emitting a 48-dim frame with the "
      f"{policy['gait_reference']['n_samples']}-sample {tables} reference tables."
    )
  heading_hold = heading_hold_metadata(env)
  if heading_hold is not None:
    policy["heading_hold"] = heading_hold
    print(f"[INFO]: Stamping heading_hold contract: {heading_hold}")

  # Parity check on real observations from the env.
  obs, _ = env.reset()
  x = obs["actor"]
  assert isinstance(x, torch.Tensor)
  frame = policy["single_observation_size"]
  with torch.no_grad():
    expected = actor.mlp(actor.obs_normalizer(x)).cpu().numpy()
  got = json_forward(policy, to_robot_order(x, x.shape[-1] // frame, frame))
  per_env = np.abs(got - expected).max(axis=1)
  err = float(per_env.max())
  print(
    f"[INFO]: parity max_abs_err = {err:.2e} "
    f"(median {float(np.median(per_env)):.2e} over {len(per_env)} envs, "
    f"action magnitude {float(np.abs(expected).max()):.2f})"
  )
  if err >= cfg.parity_tol:
    raise RuntimeError(f"Export parity FAILED ({err:.2e}) -- do not deploy.")

  out = Path(cfg.output)
  out.write_text(json.dumps(policy))
  print(f"[INFO]: Wrote {out} (in_shape: {policy['in_shape']})")

  if cfg.golden_output is not None:
    if not is_gait:
      raise ValueError("`golden_output` only applies to gait tasks.")
    golden_path = Path(cfg.golden_output)
    if kind == "jump":
      golden = make_jump_golden(env, policy["jump_reference"])
    else:
      golden = make_gait_golden(env, policy["gait_reference"])
      if kind == "mixed_jump":
        golden["jump_slot"] = policy["jump_slot"]
        golden["slot_cases"] = make_slot_golden(
          env, policy["gait_reference"], policy["jump_slot"]
        )
    golden_path.write_text(json.dumps(golden))
    print(f"[INFO]: Wrote {golden_path} (deploy parity fixture)")

  env.close()

  if cfg.upload_wandb:
    if cfg.wandb_run_path is None:
      raise ValueError("`upload_wandb` requires `wandb_run_path`.")
    import wandb

    run = wandb.Api().run(str(cfg.wandb_run_path))
    run.upload_file(str(out), root=str(out.parent))
    print(f"[INFO]: Uploaded {out.name} to W&B run {cfg.wandb_run_path}")

  return out


def main() -> None:
  maybe_print_top_level_help("export-pupper-policy")

  import mjlab
  import mjlab.tasks  # noqa: F401

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    ExportConfig,
    args=remaining_args,
    default=ExportConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
