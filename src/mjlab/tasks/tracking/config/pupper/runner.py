"""Trick runner: puts the robot's trick policy JSON on the W&B run as it trains.

Like the Pupper walking runner (``mjlab.tasks.pupper.rl.runner``): at every
checkpoint save, and once more on Ctrl+C, the current policy is exported with
``export_trick.py``'s exporter, parity-checked against the live actor, and
uploaded to the run's Files as ``policy.json``. On the robot, gemini-pupper's
``robot/get_trick.py <entity>/<project>/<run-id> --name <trick>`` fetches it.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import wandb

from mjlab.rl.runner import wandb_logging_active
from mjlab.tasks.pupper.export import json_forward
from mjlab.tasks.tracking.config.pupper.export_trick import export_trick_policy_from_env
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends

DEPLOY_JSON_NAME = "policy.json"
_PARITY_TOL = 1e-3
_PARITY_MAX_ENVS = 512


class PupperTrickRunner(MotionTrackingOnPolicyRunner):
  """Tracking runner that emits the robot's trick policy JSON at every checkpoint."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # The base runner clears registry_name once it has logged the artifact use.
    self._motion_artifact = self.registry_name

  def save(self, path: str, infos=None):
    super().save(path, infos)
    self._try_export()

  def learn(self, *args, **kwargs):
    try:
      return super().learn(*args, **kwargs)
    except KeyboardInterrupt:
      print("\n[INFO] Interrupted -- exporting the current trick policy before exit.")
      self._try_export()
      raise

  def trick_name(self) -> str:
    """The W&B motion artifact's name if training pulled one, else the motion file's."""
    if self._motion_artifact:
      return self._motion_artifact.split("/")[-1].split(":")[0]
    cmd = self.env.unwrapped.command_manager.get_term("motion")
    assert isinstance(cmd, MotionCommand)
    return Path(cmd.cfg.motion_file).stem

  def _try_export(self) -> None:
    try:
      log_dir = getattr(self.logger, "log_dir", None)
      if log_dir is None:
        print("[WARN] No log dir; skipping the trick policy export.")
        return
      self._export(Path(log_dir))
    except Exception as e:  # noqa: BLE001 - a failed export must not fail the run.
      print(f"[WARN] Trick policy export failed: {e}")

  def _export(self, out_dir: Path) -> None:
    env = self.env.unwrapped
    configure_torch_backends(allow_tf32=False)  # parity is checked in float64
    self.alg.eval_mode()
    try:
      actor = self.alg.get_policy()
      policy = export_trick_policy_from_env(actor, env, self.trick_name())
      obs = env.observation_manager.compute()["actor"]
      assert isinstance(obs, torch.Tensor)
      obs = obs[:_PARITY_MAX_ENVS]
      with torch.no_grad():
        expected = actor.mlp(actor.obs_normalizer(obs)).cpu().numpy()
      err = float(abs(json_forward(policy, obs.cpu().numpy()) - expected).max())
      if err >= _PARITY_TOL:
        raise RuntimeError(f"export parity failed ({err:.2e}); not writing")
    finally:
      self.alg.train_mode()
      configure_torch_backends(allow_tf32=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / DEPLOY_JSON_NAME
    out.write_text(json.dumps(policy))
    if wandb_logging_active(self.logger) and self.cfg["upload_model"] and wandb.run:
      run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
      wandb.Api().run(run_path).upload_file(str(out), root=str(out_dir))
      print(
        f"[INFO] Trick policy '{policy['trick']['name']}' replaced on W&B Files as "
        f"{DEPLOY_JSON_NAME} (parity {err:.1e}) -- on the robot: "
        f"python3 robot/get_trick.py {run_path} --name {policy['trick']['name']}"
      )
