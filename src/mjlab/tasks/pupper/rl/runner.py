"""Pupper runner: writes the on-robot deploy JSON when the run ends.

The robot cannot convert a ``.pt`` checkpoint itself -- that needs a GPU and
MuJoCo-Warp -- so deployment used to require a manual ``export-pupper-policy``
run on the training machine after the fact. This exports ``policy.json`` to the
W&B run automatically, which is what ``pupper_gait_deploy/download_policy.py``
pulls down.

The export runs at every checkpoint save (each ``save_interval`` iterations,
plus the final save), overwriting the previous upload, and once more on a
Ctrl+C -- the interrupt skips the final on-disk save, so the in-memory policy
is fresher than the last checkpoint and is exported directly. The run
therefore always carries a deployable JSON of its newest checkpointed policy;
a hard kill (``SIGKILL``) or a cluster preemption at worst loses the
iterations since the last save.

The file name is stable rather than iteration-stamped: W&B keeps the newest
upload under that name, so the deploy script never has to know which iteration
it wants.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import wandb

from mjlab.rl.runner import wandb_logging_active
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends

DEPLOY_JSON_NAME = "policy.json"

# Above this the export is almost certainly wrong (wrong frame layout, stale
# normalizer) and shipping it would be worse than shipping nothing. Matches the
# ``export-pupper-policy`` script's default, and must be read together with the
# TF32 handling in _export_deploy_json: the JSON forward pass is float64 while the
# torch reference is float32, so the floor is set by the reference's precision.
_PARITY_TOL = 1e-3

# Parity-check batch cap. The check is a float64 numpy forward pass over real
# observations; at training scale (16k envs) that is seconds per call, which is
# fine once per run but not once per checkpoint. A few hundred envs pin the
# frame layout and normalizer fold just as well.
_PARITY_MAX_ENVS = 512


class PupperOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner that emits an on-robot deploy JSON at every checkpoint."""

  def save(self, path: str, infos=None):
    super().save(path, infos)
    self._try_export()

  def learn(self, *args, **kwargs):
    try:
      result = super().learn(*args, **kwargs)
    except KeyboardInterrupt:
      # The final checkpoint save inside learn() is skipped on interrupt, so the
      # in-memory policy is the freshest thing available -- newer than the last
      # interval checkpoint on disk. Export that rather than reloading the .pt.
      print("\n[INFO] Interrupted -- exporting the current policy before exit.")
      self._try_export()
      raise
    return result

  def _try_export(self) -> None:
    """Export and upload, swallowing failures so they cannot mask a finished run."""
    try:
      log_dir = getattr(self.logger, "log_dir", None)
      if log_dir is None:
        print("[WARN] No log dir; skipping Pupper deploy JSON export.")
        return
      self._export_deploy_json(Path(log_dir))
    except Exception as e:  # noqa: BLE001 - a failed export must not fail the run.
      print(f"[WARN] Pupper deploy JSON export failed: {e}")

  def _export_deploy_json(self, out_dir: Path) -> None:
    # Local imports: this module is loaded by the task registry, and the export
    # helpers pull in the gait reference tables, which are only needed here.
    from mjlab.tasks.pupper.export import (
      export_pupper_policy_from_env,
      json_forward,
      to_robot_order,
    )
    from mjlab.tasks.pupper_gait.export import (
      GAIT_OBS_COMPONENTS,
      GAIT_SINGLE_OBS_DIM,
      gait_frame_kind,
      gait_reference_metadata,
      heading_hold_metadata,
      jump_reference_metadata,
      jump_slot_metadata,
    )

    env = self.env.unwrapped
    # Training runs with TF32 matmuls (10-bit mantissa), but the parity check
    # compares the torch actor against a float64 re-implementation of the same
    # forward pass -- so under TF32 the *reference* is the imprecise side, and the
    # disagreement lands around 5e-4, well past the tolerance. The standalone
    # exporter sidesteps this by calling configure_torch_backends(allow_tf32=False)
    # at startup. Do the same here for the duration of the export and restore
    # afterwards, since training wants TF32 back for speed.
    configure_torch_backends(allow_tf32=False)
    # ``get_inference_policy`` calls ``alg.eval_mode()`` and never restores it,
    # which would freeze the observation normalizer's running statistics for the
    # rest of training. Switch explicitly and restore in the finally block; eval
    # mode also stops the parity forward pass from feeding its batch into those
    # statistics.
    self.alg.eval_mode()
    try:
      actor = self.alg.get_policy()

      kind = gait_frame_kind(env)
      policy = export_pupper_policy_from_env(
        actor,
        env,
        single_obs_dim=GAIT_SINGLE_OBS_DIM if kind else None,
        obs_components=GAIT_OBS_COMPONENTS if kind else None,
      )
      if kind == "jump":
        policy["jump_reference"] = jump_reference_metadata(env)
      elif kind == "mixed_jump":
        policy["gait_reference"] = gait_reference_metadata(env)
        policy["jump_slot"] = jump_slot_metadata(env)
      elif kind is not None:
        policy["gait_reference"] = gait_reference_metadata(env)
      heading_hold = heading_hold_metadata(env)
      if heading_hold is not None:
        policy["heading_hold"] = heading_hold

      # Parity against the live actor on real observations, capped to
      # _PARITY_MAX_ENVS since this now runs at every checkpoint. The JSON is a
      # hand-rolled re-implementation of the forward pass in the robot's layout,
      # so this is the check that it actually matches what was trained. Note the
      # extra compute() nudges stateful obs terms (IMU latency buffer, history)
      # by one frame once per save interval -- noise well under the domain
      # randomization already applied to the same quantities.
      obs = env.observation_manager.compute()["actor"]
      assert isinstance(obs, torch.Tensor)
      obs = obs[:_PARITY_MAX_ENVS]
      frame = policy["single_observation_size"]
      with torch.no_grad():
        expected = actor.mlp(actor.obs_normalizer(obs)).cpu().numpy()
      got = json_forward(policy, to_robot_order(obs, obs.shape[-1] // frame, frame))
      err = float(abs(got - expected).max())
      if err >= _PARITY_TOL:
        raise RuntimeError(f"export parity failed ({err:.2e}); not writing")
    finally:
      self.alg.train_mode()
      configure_torch_backends(allow_tf32=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / DEPLOY_JSON_NAME
    out.write_text(json.dumps(policy))

    if wandb_logging_active(self.logger) and self.cfg["upload_model"] and wandb.run:
      # Upload through the public API rather than wandb.save: save(policy="now")
      # uploads a registered path exactly once, so with per-checkpoint exports
      # the run's policy.json silently stayed the FIRST checkpoint's version
      # (observed on run tryieau1 -- policy.json frozen minutes behind the
      # newest model_*.pt). Api().upload_file overwrites the server-side file
      # on every call, synchronously, which also makes it safe on the Ctrl+C
      # path where the process exits right after.
      run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
      wandb.Api().run(run_path).upload_file(str(out), root=str(out_dir))
      print(
        f"[INFO] Deploy JSON replaced on W&B Files as {DEPLOY_JSON_NAME} "
        f"(parity {err:.1e}) -- fetch with: "
        f"python3 download_policy.py {run_path}"
      )
