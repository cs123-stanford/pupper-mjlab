"""Live mid-training viewer for the CS123 Colab notebook.

Launched as a *background process* by the notebook right before the training
cell, so the viewer keeps serving while the training cell blocks. Pure add-on:
mirrors the notebook's "Watch it" cell plus play.py's checkpoint hot-swap
wiring, with two lab-specific behaviors:

- Waits for the first checkpoint of a *fresh* run (one written after this
  process started), so relaunching before each training run always latches
  onto the run that follows.
- Auto-pauses whenever no browser is connected and resumes on connect. A
  paused viewer skips env.step() entirely, so training only pays the
  shared-GPU tax while someone is actually watching.

Usage:
  python notebooks/watch_live.py <TASK> [--url-file PATH] [--log-root DIR]
      [--any-checkpoint]
"""

import argparse
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch
import viser

import mjlab.tasks  # noqa: F401  (register tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def wait_for_checkpoint(log_root: Path, after: float) -> Path:
  """Block until a model_*.pt written after `after` exists; return the newest."""
  while True:
    ckpts = [p for p in log_root.glob("*/*/model_*.pt") if p.stat().st_mtime >= after]
    if ckpts:
      run_dir = max({p.parent for p in ckpts}, key=lambda d: d.stat().st_mtime)
      in_run = sorted(
        run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])
      )
      if in_run:
        return in_run[-1]
    time.sleep(10.0)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  parser.add_argument("--url-file", type=Path, default=None)
  parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
  parser.add_argument(
    "--any-checkpoint",
    action="store_true",
    help="Latch onto the newest existing checkpoint instead of waiting for a "
    "fresh one (useful outside the launch-then-train notebook flow).",
  )
  args = parser.parse_args()
  start_time = 0.0 if args.any_checkpoint else time.time()

  # Server + share URL first: the link is ready within seconds, the scene
  # populates once the first checkpoint lands.
  server = viser.ViserServer(label="pupper-live")
  url = server.request_share_url()
  print("Open the viewer here:", url, flush=True)
  if args.url_file is not None:
    args.url_file.write_text(url)
  waiting_note = server.gui.add_markdown(
    "**Waiting for the first checkpoint** — the robot appears about a minute "
    "into training."
  )

  resume_path = wait_for_checkpoint(args.log_root, after=start_time)
  ckpt_dir = resume_path.parent
  print("First checkpoint:", resume_path, flush=True)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device),
    clip_actions=agent_cfg.clip_actions,
  )
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)

  def _reload_policy(path: str):
    runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)

  policy = _reload_policy(str(resume_path))

  def fetch_available() -> list[tuple[str, str]]:
    now = time.time()
    entries: list[tuple[str, str, int]] = []
    for f in sorted(ckpt_dir.glob("*.pt")):
      try:
        step = int(f.stem.split("_")[1])
      except (IndexError, ValueError):
        step = 0
      ago = format_time_ago(int(now - f.stat().st_mtime))
      entries.append((f.name, ago, step))
    entries.sort(key=lambda x: x[2])
    return [(name, t) for name, t, _ in entries]

  ckpt_manager = CheckpointManager(
    current_name=resume_path.name,
    fetch_available=fetch_available,
    load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
  )

  waiting_note.remove()
  viewer = ViserPlayViewer(
    env, policy, viser_server=server, checkpoint_manager=ckpt_manager
  )

  def idle_monitor() -> None:
    """Pause when no browser is connected; resume (only our pause) on connect."""
    auto_paused = False
    while True:
      time.sleep(2.0)
      try:
        connected = len(server.get_clients())
      except Exception:
        continue
      if connected == 0 and not viewer._is_paused:
        viewer.request_toggle_pause()
        auto_paused = True
        print("[watch_live] no viewers connected -- paused", flush=True)
      elif connected > 0 and viewer._is_paused and auto_paused:
        viewer.request_toggle_pause()
        auto_paused = False
        print("[watch_live] viewer connected -- resumed", flush=True)

  threading.Thread(target=idle_monitor, daemon=True).start()
  viewer.run()


if __name__ == "__main__":
  main()
