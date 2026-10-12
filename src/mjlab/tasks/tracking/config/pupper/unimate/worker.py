"""UniMate text-to-motion worker: runs inside UniMate's own environment.

    <UniMate>/.venv/bin/python worker.py --unimate <UniMate> [--exp ...] [--step ...]

Started and spoken to by ``UniMate`` in ``__init__.py`` (in the mjlab
environment); you don't run it by hand. It imports only UniMate, never mjlab.

Keeps the model loaded and answers JSON lines on stdin with JSON lines on stdout:

  {"cmd": "register", "name": ..., "names": [...], "clean": [...], "parents": [...],
   "rest": [[x, y, z], ...], "face": ["<right hip joint>", "<left hip joint>"]}
      rest: global rest positions in the robot frame (x forward, y left, z up), metres.
  {"cmd": "generate", "skeleton": ..., "prompt": ..., "cfg": 3.0, "seed": 0, "n": 1,
   "out": "path.npz"[, "prompts": [...], "overlap": 10]}
      "prompts" (two or more) chains 2 s segments with UniMate's motion expansion:
      each segment after the first starts pinned to the previous one's last
      "overlap" frames, so the motion is continuous across prompts.
      writes positions (n, 60, J, 3) in the same frame and units, 30 fps, joints in
      the registered order.

The skeleton conditioning is built exactly as UniMate's stage-4 feature extraction
and dataset loader build it (process_tpose -> build_topology_cond -> T-pose
condition), so no dataset features are needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
# stdout carries the JSON protocol only; everything libraries print goes to stderr.
PROTO = sys.stdout
sys.stdout = sys.stderr
ARGS = argparse.ArgumentParser()
ARGS.add_argument(
  "--unimate", required=True, help="the UniMate checkout (install.sh's DIR)"
)
ARGS.add_argument("--exp", default="unimate_uniml3d_f60_v3")
ARGS.add_argument("--step", type=int, default=150000)
ARGS.add_argument(
  "--stats", default="truebones", help="normalization: the animals' stats"
)
OPTS = ARGS.parse_args()
UNIMATE = os.path.abspath(OPTS.unimate)
sys.path.insert(0, UNIMATE)
os.chdir(UNIMATE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from Animation import (  # noqa: E402
  offsets_from_positions,
  positions_global,
  rotations_global,
)
from data_process.utils.motion_features import (  # noqa: E402
  build_topology_cond,
  process_tpose,
)
from Quaternions import Quaternions  # noqa: E402
from unimate.configs.schema import MainConfig  # noqa: E402
from unimate.dataset.mixture.collate import mixture_batch_collate  # noqa: E402
from unimate.dataset.transforms import (  # noqa: E402
  apply_normalization,
  build_parent_features,
)
from unimate.inference.generate import generate_samples  # noqa: E402
from unimate.inference.motion_expansion import expand_motion_chain  # noqa: E402
from unimate.models.factory import create_model, create_transport  # noqa: E402
from unimate.models.flow.transport import Sampler  # noqa: E402
from unimate.models.text_encoder.factory import create_text_encoder  # noqa: E402
from unimate.training.ema import EMAModel  # noqa: E402
from unimate.utils.motion_utils import (  # noqa: E402
  recover_unimate_joint_pos_from_ric,
  recover_unimate_joint_pos_from_rot,
)
from unimate.utils.text_emb_cache import pool, sequences_from_hidden  # noqa: E402

EXP = os.path.join(UNIMATE, "outputs", OPTS.exp)
STATS_DATASET = OPTS.stats


# Robot frame (x fwd, y left, z up) <-> UniMate/BVH raw frame (X left, Y up, Z fwd).
def to_raw(p):
  p = np.asarray(p, float)
  return np.stack([p[..., 1], p[..., 2], p[..., 0]], -1)


def from_raw(p):
  p = np.asarray(p, float)
  return np.stack([p[..., 2], p[..., 0], p[..., 1]], -1)


def smooth_seams(pos, seams, width=10):
  """Remove the jump at each chain seam and fade the correction out over `width` frames.

  pos (n, T, J, 3). The jump is the seam step minus the local velocity (the mean of
  the steps just before and after), so ordinary motion across the seam is kept.
  """
  pos = pos.copy()
  for s in seams:
    if s < 2 or s + 1 >= pos.shape[1]:
      continue
    expected = 0.5 * ((pos[:, s - 1] - pos[:, s - 2]) + (pos[:, s + 1] - pos[:, s]))
    jump = (pos[:, s] - pos[:, s - 1]) - expected  # (n, J, 3)
    w = min(width, pos.shape[1] - s)
    ramp = 1.0 - np.arange(w) / w  # 1 at the seam -> 0
    pos[:, s : s + w] -= ramp[None, :, None, None] * jump[:, None]
  return pos


def log(*a):
  print(*a, file=sys.stderr, flush=True)


class Worker:
  def __init__(self):
    t0 = time.time()
    self.config = MainConfig.from_json(os.path.join(EXP, "config.json"))
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = create_model(
      dataset_config=self.config.dataset, model_config=self.config.model
    )
    ckpt = os.path.join(EXP, "checkpoints", f"checkpoint_step_{OPTS.step}.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    self.model.load_state_dict(state.get("model_state_dict", state))
    if self.config.training.use_ema and "ema_state_dict" in state:
      ema = EMAModel(
        parameters=self.model.parameters(),
        decay=self.config.training.ema_decay,
        use_ema_warmup=True,
      )
      ema.load_state_dict(state["ema_state_dict"])
      ema.copy_to(self.model.parameters())
    del state
    self.model.to(self.device).eval()
    self.transport = create_transport(training_config=self.config.training)
    self.sampler = Sampler(self.transport)
    self.encoder = create_text_encoder(
      encoder_type=self.config.model.text_encoder_type,
      encoder_version=self.config.model.text_encoder_version,
      device=str(self.device),
      pool=False,
    )
    stats = np.load(os.path.join(EXP, "dataset_stats.npy"), allow_pickle=True).item()[
      STATS_DATASET
    ]
    self.stats = stats
    self.skeletons = {}
    log(f"[unimate] ready in {time.time() - t0:.1f}s on {self.device}")

  # ---- text ------------------------------------------------------------------
  def encode(self, text):
    with torch.no_grad():
      inputs = self.encoder.tokenize(text)
      hidden = self.encoder(inputs)
    tokens = sequences_from_hidden(
      hidden.detach().cpu(), inputs["attention_mask"].cpu()
    )[0]
    return {"caption_emb": pool(tokens), "caption_tokens": tokens}

  # ---- skeleton --------------------------------------------------------------
  def register(self, req):
    names = list(req["names"])
    clean = list(req["clean"])
    parents = [int(p) for p in req["parents"]]
    rest_raw = to_raw(req["rest"])  # global rest positions
    local = rest_raw.copy()
    for j, p in enumerate(parents):
      if p >= 0:
        local[j] = rest_raw[j] - rest_raw[p]
    J = len(names)
    tpos_data = {
      "names": np.array(names),
      "fps": np.array(30.0),
      "parents": np.array(parents),
      "rest_local_rot": np.tile([1.0, 0.0, 0.0, 0.0], (J, 1)),
      "rest_local_pos": local,
    }
    face = {"r_hip": {"raw": req["face"][0]}, "l_hip": {"raw": req["face"][1]}}
    (
      tpos_anim,
      offsets,
      scale_factor,
      _ground,
      parents_b,
      names_b,
      _fps,
      bfs_order,
      face_idxs,
      body_axis,
    ) = process_tpose(tpos_data, face_joints=face, target_diameter=2.0)
    clean_b = [clean[i] for i in bfs_order]
    tpos_global = positions_global(tpos_anim)[0]
    cond = build_topology_cond(
      req["name"],
      parents_b,
      offsets,
      names_b,
      clean_b,
      tpos_global,
      tpos_anim.rotations.qs[0],
      rotations_global(tpos_anim).qs[0],
      face_joint_idxs=face_idxs,
      body_axis=body_axis,
      scale_factor=scale_factor,
    )
    parents_arr = np.asarray(parents_b)
    tpos = np.asarray(cond["tpos_first_frame"]).copy()
    if self.config.dataset.ground_motion_height:
      tpos[..., 1] -= tpos[..., 1].min()
    # Joint-name embeddings: one pooled vector per cleaned name.
    joint_names_emb = np.stack(
      [np.asarray(self.encode(n)["caption_emb"]) for n in clean_b]
    )
    self.skeletons[req["name"]] = dict(
      names=names_b,
      clean=clean_b,
      parents=parents_arr,
      bfs_order=list(bfs_order),
      tpos=tpos,
      offsets=offsets_from_positions(tpos, parents_arr),
      scale=float(scale_factor),
      cond=cond,
      joint_names_emb=joint_names_emb,
      # where the canonical frame's origin/facing sits relative to the robot frame
      rest_raw=rest_raw,
    )
    return {"ok": True, "joints": J, "scale": float(scale_factor), "order": names_b}

  def _batch(self, sk, caption_enc):
    cfg = self.config.dataset
    J = len(sk["names"])
    tpos = sk["tpos"]
    identity_6d = Quaternions.id(1).rotation_matrix(cont6d=True)[0]
    pad = np.zeros((J, cfg.feature_len - 3))
    pad[:, :6] = identity_6d
    tpos12 = np.concatenate([tpos, pad], -1)
    mean = np.zeros((J, cfg.feature_len))
    std = np.zeros((J, cfg.feature_len))
    mean[0], std[0] = self.stats["mean_root"], self.stats["std_root"]
    mean[1:], std[1:] = self.stats["mean_local"], self.stats["std_local"]
    tpos_n = apply_normalization(tpos12, mean, std)
    motion = np.zeros((cfg.max_motion_length, J, cfg.feature_len))
    c = sk["cond"]
    b = {
      "motion": motion,
      "max_motion_length": cfg.max_motion_length,
      "motion_length": cfg.max_motion_length,
      "max_joints": cfg.max_joints,
      "parents": sk["parents"],
      "edge_indexs": np.asarray(c["edge_indexs"]),
      "tpos_first_frame": tpos_n,
      "tpos_first_frame_parents": build_parent_features(tpos_n, sk["parents"])[
        "tpos_first_frame_parents"
      ],
      "offsets": sk["offsets"],
      "joint_graph_dist": np.asarray(c["joint_graph_dists"]),
      "joint_relations": np.asarray(c["joint_relations"]),
      "joint_depths": np.asarray(c["joint_depths"]),
      "spectral_feats": np.asarray(c["spectral_feats"]),
      "joint_names_emb": sk["joint_names_emb"],
      "object_type": "pupper",
      "start_idx": 0,
      "mean": mean,
      "std": std,
      "split_tag": "eval",
    }
    b.update(caption_enc)
    return b

  def generate(self, req):
    sk = self.skeletons[req["skeleton"]]
    n = int(req.get("n", 1))
    if req.get("seed") is not None:
      torch.manual_seed(int(req["seed"]))
      np.random.seed(int(req["seed"]))
    prompts = [p for p in (req.get("prompts") or [req["prompt"]]) if p.strip()]
    conds = []
    for text in prompts:
      enc = self.encode(text)
      _motion, cond = mixture_batch_collate([self._batch(sk, enc) for _ in range(n)])
      conds.append(
        {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in cond.items()}
      )
    cond = conds[0]
    cfg = self.config.dataset
    shape = (n, cfg.max_joints, cfg.feature_len, cfg.max_motion_length)
    cfg_scale = float(req.get("cfg", self.config.sampling.cfg_scale))
    t0 = time.time()
    with torch.no_grad():
      if len(conds) == 1:
        samples = generate_samples(
          model=self.model,
          cond=cond,
          motion_shape=shape,
          diff_model="flow",
          diffusion=self.transport,
          gen_diffusion=self.sampler,
          device=self.device,
          cfg_scale=cfg_scale,
        )
      else:
        samples = expand_motion_chain(
          cond_per_segment=conds,
          motion_shape=shape,
          overlap=int(req.get("overlap", 10)),
          sample_model=self.model,
          diff_model="flow",
          diffusion=self.transport,
          gen_diffusion=self.sampler,
          cfg_scale=cfg_scale,
          device=self.device,
        )
    gen_s = time.time() - t0
    J = len(sk["names"])
    out = []
    for i in range(n):
      m = samples[i, :J].cpu().permute(2, 0, 1).numpy()  # (T, J, D)
      mean = cond["mean"][i, :J].cpu().numpy()[None]
      std = cond["std"][i, :J].cpu().numpy()[None]
      m = m * std + mean
      if req.get("decode", "fk") == "ric":
        pos = recover_unimate_joint_pos_from_ric(m)
      else:
        pos = recover_unimate_joint_pos_from_rot(m, sk["parents"], sk["offsets"])
      out.append(pos)
    pos = np.stack(out)  # canonical: Y up, facing +Z, diameter 2
    pos = from_raw(pos / sk["scale"])  # back to metres, robot frame
    # back to the registered joint order
    inv = np.argsort(sk["bfs_order"])
    pos = pos[:, :, inv]
    T = cfg.max_motion_length
    overlap = int(req.get("overlap", 10))
    seams = [T + k * (T - overlap) for k in range(len(prompts) - 1)]
    pos = smooth_seams(pos, seams)
    np.savez(
      req["out"],
      positions=pos,
      fps=30.0,
      prompt=" | ".join(prompts),
      seams=np.array(seams),
    )
    return {
      "ok": True,
      "out": req["out"],
      "seconds": round(gen_s, 2),
      "shape": list(pos.shape),
    }


def main():
  w = Worker()
  print(json.dumps({"ok": True, "ready": True}), file=PROTO, flush=True)
  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    try:
      req = json.loads(line)
      res = getattr(w, req["cmd"])(req)
    except Exception as e:  # noqa: BLE001
      import traceback

      traceback.print_exc(file=sys.stderr)
      res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(res), file=PROTO, flush=True)


if __name__ == "__main__":
  main()
