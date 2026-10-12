"""Text to Pupper motion with UniMate (SIGGRAPH Asia 2026, Mou et al.).

UniMate generates motion for any skeleton from a sentence: 2 s (60 frames at
30 fps) per prompt, and longer motions by chaining prompts. It runs in its own
environment (``install.sh``) as a worker process (``worker.py``); this module
starts it, describes Pupper's skeleton to it, and turns what it generates into a
tracking-task motion:

    um = UniMate("/content/UniMate")              # loads the model (about a minute)
    raw = um.generate(["An object sits down."], n=4)   # (4, 60, 19, 3) joint positions
    fit = fit_pupper(raw[0])                       # IK -> md.Motion at 50 Hz, + stats
    fit.motion.save_npz("sit.npz")

The fit is kinematic: no physics. Check it with ``play_motion`` before training.
UniMate's code is MIT licensed; its released checkpoints are CC BY-NC 4.0
(non-commercial use).
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import threading
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.env_cfgs import get_spec_with_hulls

HERE = Path(__file__).resolve().parent
GEN_FPS = 30.0  # UniMate's frame rate
SEGMENT = 60  # frames per prompt
FPS = 50.0  # Pupper motions

# ----------------------------------------------------------------------------- skeleton
# Pupper's own geometry as an animal rig (robot frame: x forward, y left, z up;
# metres): hips and shoulders, knees and feet where the robot has them at its
# default stand, plus a virtual spine, neck and head so the rig reads as a dog.
# Joint names use the words UniMate's animal rigs were labelled with (front legs:
# Shoulder / Upper Arm / Forearm / Hand; hind legs: Thigh / Shin / Foot).
H = 0.161  # hip and shoulder height at the default stand
KNEE_X, KNEE_Z = 0.0535, 0.0956  # knees sit 5.35 cm behind their hips
FOOT_Z = 0.0196  # foot sphere centre


def pupper_skeleton() -> dict:
  j = []  # (name, clean name, parent, position)

  def add(name, clean, parent, pos):
    j.append((name, clean, parent, np.array(pos, float)))
    return len(j) - 1

  hips = add("pelvis", "Hips", -1, (-0.075, 0.0, H))
  spine = add("spine", "Spine", hips, (0.0, 0.0, H))
  chest = add("chest", "Spine", spine, (0.075, 0.0, H))
  neck = add("neck", "Neck", chest, (0.115, 0.0, H + 0.025))
  add("head", "Head", neck, (0.155, 0.0, H + 0.04))
  for side, sy in (("Right", -1), ("Left", 1)):
    t = add(f"{side.lower()}_thigh", f"{side} Thigh", hips, (-0.075, sy * 0.0725, H))
    s = add(
      f"{side.lower()}_shin", f"{side} Shin", t, (-0.075 - KNEE_X, sy * 0.0725, KNEE_Z)
    )
    add(f"{side.lower()}_foot", f"{side} Foot", s, (-0.0843, sy * 0.0905, FOOT_Z))
  for side, sy in (("Right", -1), ("Left", 1)):
    sh = add(
      f"{side.lower()}_shoulder", f"{side} Shoulder", chest, (0.075, sy * 0.04, H)
    )
    ua = add(
      f"{side.lower()}_upper_arm", f"{side} Upper Arm", sh, (0.075, sy * 0.0835, H)
    )
    fa = add(
      f"{side.lower()}_forearm",
      f"{side} Forearm",
      ua,
      (0.075 - KNEE_X, sy * 0.0835, KNEE_Z),
    )
    add(f"{side.lower()}_hand", f"{side} Hand", fa, (0.0657, sy * 0.1015, FOOT_Z))
  names, clean, parents, rest = zip(*j, strict=True)
  return dict(
    name="pupper",
    names=list(names),
    clean=list(clean),
    parents=list(parents),
    rest=np.stack(rest).tolist(),
    face=["right_thigh", "left_thigh"],
    # leg -> (leg root joint, foot joint), for the IK
    legs={
      "front_r": ("right_upper_arm", "right_hand"),
      "front_l": ("left_upper_arm", "left_hand"),
      "back_r": ("right_thigh", "right_foot"),
      "back_l": ("left_thigh", "left_foot"),
    },
  )


SKELETON = pupper_skeleton()


# ----------------------------------------------------------------------------- the worker
class UniMate:
  """The UniMate worker process: one JSON line in, one JSON line out."""

  def __init__(
    self,
    unimate_dir: str | Path = "UniMate",
    exp: str = "unimate_uniml3d_f60_v3",
    step: int = 150000,
    log: str | Path | None = None,
  ):
    self.dir = Path(unimate_dir).resolve()
    self.log = Path(log) if log else self.dir / "worker.log"
    python = self.dir / ".venv" / "bin" / "python"
    if not python.exists():
      raise FileNotFoundError(f"{python} is missing: run install.sh {self.dir} first")
    cmd = [str(python), "-u", str(HERE / "worker.py"), "--unimate", str(self.dir)]
    cmd += ["--exp", exp, "--step", str(step)]
    # UniMate's own environment, not the notebook kernel's: Jupyter's
    # MPLBACKEND (matplotlib_inline) and PYTHONPATH would break its imports.
    env = {
      k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")
    }
    env["MPLBACKEND"] = "Agg"
    self.proc = subprocess.Popen(
      cmd,
      env=env,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=self.log.open("w"),
      text=True,
      bufsize=1,
    )
    self.lock = threading.Lock()
    self._read()  # {"ready": true} once the model is loaded
    sk = SKELETON
    self.call(
      cmd="register",
      **{k: sk[k] for k in ("name", "names", "clean", "parents", "rest", "face")},
    )
    self.n_out = 0

  def _read(self) -> dict:
    assert self.proc.stdout is not None
    line = self.proc.stdout.readline()
    if not line:
      tail = "\n".join(self.log.read_text().splitlines()[-15:])
      raise RuntimeError(f"the UniMate worker stopped; the end of {self.log}:\n{tail}")
    res = json.loads(line)
    if not res.get("ok"):
      raise RuntimeError(f"UniMate: {res.get('error')} (details in {self.log})")
    return res

  def call(self, **req) -> dict:
    assert self.proc.stdin is not None
    if self.proc.poll() is not None:
      raise RuntimeError("UniMate is stopped: run the cell that starts it again")
    with self.lock:
      self.proc.stdin.write(json.dumps(req) + "\n")
      self.proc.stdin.flush()
      return self._read()

  def generate(
    self,
    prompts: list[str] | str,
    n: int = 4,
    seed: int | None = 0,
    cfg: float = 3.0,
    overlap: int = 10,
  ) -> np.ndarray:
    """Pupper-skeleton joint positions (n, T, 19, 3) at 30 fps, in metres.

    One prompt gives T = 60 (2 s). Several prompts are chained: each starts from
    the last ``overlap`` frames of the one before, so the motion is continuous.
    ``cfg`` is how strongly the prompt is followed (3 is UniMate's default; 5-7
    when a prompt is ignored, lower for calmer motion).
    """
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    self.n_out += 1
    out = self.dir / "outputs" / f"pupper_{self.n_out:03d}.npz"
    res = self.call(
      cmd="generate",
      skeleton=SKELETON["name"],
      prompt=prompts[0],
      prompts=prompts,
      n=n,
      seed=seed,
      cfg=cfg,
      overlap=overlap,
      out=str(out),
    )
    self.seconds = res["seconds"]
    return np.load(out)["positions"]

  def close(self) -> None:
    if self.proc.poll() is None:
      self.proc.kill()


# ----------------------------------------------------------------------------- timing
def time_warp(
  n_frames: int, durations: list[float], overlap: int = 10, slow: float = 1.0
):
  """Piecewise-linear map from Pupper time to UniMate time, as (out, src) knots.

  Each chained prompt is stretched on its own to last ``durations[k]`` seconds
  (nominally 2 s; prompts after the first add only their 60 - overlap new
  frames), then everything is slowed by ``slow``.
  """
  src = [0.0]
  for k in range(len(durations)):
    frames = SEGMENT if k == 0 else SEGMENT - overlap
    src.append(src[-1] + frames / GEN_FPS)
  src[-1] = (n_frames - 1) / GEN_FPS  # the chain's true end
  out = [0.0]
  for k, d in enumerate(durations):
    out.append(out[-1] + (src[k + 1] - src[k]) * (d / 2.0) * slow)
  return np.array(out), np.array(src)


def warp_resample(pos: np.ndarray, knots: tuple, fps: float = FPS) -> np.ndarray:
  out_k, src_k = knots
  t_out = np.arange(0.0, out_k[-1] + 1e-9, 1.0 / fps)
  f = np.clip(np.interp(t_out, out_k, src_k) * GEN_FPS, 0, len(pos) - 1)
  i0 = np.floor(f).astype(int)
  i1 = np.minimum(i0 + 1, len(pos) - 1)
  a = (f - i0).reshape(-1, *([1] * (pos.ndim - 1)))
  return (1 - a) * pos[i0] + a * pos[i1]


def scale_travel(pos: np.ndarray, stride: float) -> np.ndarray:
  """Shrink every joint's horizontal displacement from its start by ``stride``.

  Steps and travel shrink together, and a planted foot stays planted (its
  displacement is zero), so contacts survive; heights are untouched.
  """
  out = pos.copy()
  start = pos[:1, :, :2]
  out[..., :2] = start + stride * (pos[..., :2] - start)
  return out


# ----------------------------------------------------------------------------- the fit
W_FOOT_FLOOR = 20.0  # a foot below the floor is never an acceptable compromise
W_ROOT_POS, W_ROOT_ROT, W_GROUND, W_POSE = 0.3, 0.08, 3.0, 0.01


def solve_pupper(base, R, targets, iters: int = 40):
  """Whole-body IK for Pupper along a trajectory.

  base (T, 3) body centre, R (T, 3, 3) body axes (columns x forward, y left,
  z up), targets {leg: (T, 3)} foot positions. Feet are matched first, the body
  pose weakly, legs stay near the default stance, nothing goes below the ground.
  Returns root_pos (T, 3), root_quat (T, 4) wxyz, joint_pos (T, 12).
  """
  k = md.kinematics()
  m, d, nv = k.model, k.data, k.model.nv
  T = len(base)
  abd = np.array(
    [1, 0, 0] * 4, bool
  )  # abduction joints: keep within +-0.5 rad of the stance
  lo = np.where(abd, np.array(md.DEFAULT_POSE) - 0.5, k.lower)
  hi = np.where(abd, np.array(md.DEFAULT_POSE) + 0.5, k.upper)
  qpos = np.zeros(m.nq)
  qpos[:3] = base[0]
  mujoco.mju_mat2Quat(qpos[3:7], np.asarray(R[0]).reshape(-1))
  qpos[k.qadr] = md.DEFAULT_POSE
  jac = np.zeros((3, nv))
  jr = np.zeros((6, nv))
  jr[:3, :3] = np.eye(3)
  jr[3:, 3:6] = np.eye(3)
  jp = np.zeros((12, nv))
  jp[np.arange(12), k.dadr] = W_POSE

  def lowest_point():
    best = (1.0, 0, np.zeros(3))  # height, body, world point
    for g, pts in k.hull_points.items():
      b = m.geom_bodyid[g]
      w = d.xpos[b] + pts @ d.xmat[b].reshape(3, 3).T
      j = int(np.argmin(w[:, 2]))
      if w[j, 2] < best[0]:
        best = (w[j, 2], b, w[j])
    return best

  rp, rq, jq = np.zeros((T, 3)), np.zeros((T, 4)), np.zeros((T, 12))
  for i in range(T):
    qref = np.zeros(4)
    mujoco.mju_mat2Quat(qref, np.asarray(R[i]).reshape(-1))
    for _ in range(iters):
      d.qpos[:] = qpos
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      res, J = [], []
      for leg in md.LEGS:
        mujoco.mj_jacSite(m, d, jac, None, k.foot_site[leg])
        res.append(targets[leg][i] - d.site_xpos[k.foot_site[leg]])
        J.append(jac.copy())
      ev = np.zeros(3)
      mujoco.mju_subQuat(ev, qref, qpos[3:7])
      res.append(W_ROOT_POS * (base[i] - qpos[:3]))
      J.append(W_ROOT_POS * jr[:3])
      res.append(W_ROOT_ROT * ev)
      J.append(W_ROOT_ROT * jr[3:])
      zlow, b, pw = lowest_point()
      if zlow < 0:
        mujoco.mj_jac(m, d, jac, None, pw, b)
        res.append(np.array([W_GROUND * -zlow]))
        J.append(W_GROUND * jac[2:3].copy())
      for leg in md.LEGS:  # feet stay on or above the floor, even out of reach
        fz = d.site_xpos[k.foot_site[leg]][2] - md.FOOT_RADIUS
        if fz < 0:
          mujoco.mj_jacSite(m, d, jac, None, k.foot_site[leg])
          res.append(np.array([W_FOOT_FLOOR * -fz]))
          J.append(W_FOOT_FLOOR * jac[2:3].copy())
      res.append(W_POSE * (np.array(md.DEFAULT_POSE) - qpos[k.qadr]))
      J.append(jp)
      r = np.concatenate(res)
      Jm = np.concatenate(J)
      dq = np.asarray(
        Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(len(r)), r), np.float64
      )
      mujoco.mj_integratePos(m, qpos, dq, 1.0)
      qpos[k.qadr] = np.clip(qpos[k.qadr], lo, hi)
      if np.linalg.norm(dq) < 1e-5:
        break
    rp[i], rq[i], jq[i] = qpos[:3], qpos[3:7], qpos[k.qadr]
  return rp, rq, jq


@dataclasses.dataclass
class Fit:
  motion: md.Motion  # Pupper at 50 Hz
  skeleton: np.ndarray  # (T, 19, 3) the generated skeleton, on the same clock
  feet: dict  # leg -> (T, 3) foot targets
  stats: dict

  def __str__(self) -> str:
    return "\n".join(f"{k:36} {v}" for k, v in self.stats.items())


def fit_pupper(
  raw: np.ndarray,
  durations: list[float] | None = None,
  overlap: int = 10,
  stride: float = 0.6,
  slow: float = 1.0,
  trim_start: float = 0.0,
  trim_end: float = 0.0,
  hold: float = 0.0,
) -> Fit:
  """One UniMate sample (T, 19, 3) at 30 fps -> a Pupper motion at 50 Hz.

  In order: ``stride`` shrinks steps and travel (UniMate's animals cover more
  ground than Pupper can), ``durations`` sets each chained prompt's length in
  seconds (default 2 s each) and ``slow`` stretches it all, ``trim_start`` /
  ``trim_end`` cut seconds off either end, and ``hold`` keeps the last pose
  that many seconds (to stay seated, say). Then IK fits Pupper to it.
  """
  n_prompts = 1 + max(0, round((len(raw) - SEGMENT) / (SEGMENT - overlap)))
  durations = list(durations or [2.0] * n_prompts)
  if len(durations) != n_prompts:
    raise ValueError(f"{len(durations)} durations for {n_prompts} chained prompts")
  pos = warp_resample(
    scale_travel(raw, stride), time_warp(len(raw), durations, overlap, slow)
  )
  a = int(round(trim_start * FPS))
  b = len(pos) - int(round(trim_end * FPS))
  if b - a < 2:
    raise ValueError("trimmed away the whole motion")
  pos = pos[a:b]
  if hold > 0:
    pos = np.concatenate([pos, np.repeat(pos[-1:], int(round(hold * FPS)), axis=0)])

  ix = {n: i for i, n in enumerate(SKELETON["names"])}
  pos[..., 2] += FOOT_Z  # UniMate puts the lowest joint (a foot centre) at 0
  roots = {leg: pos[:, ix[r]] for leg, (r, _) in SKELETON["legs"].items()}
  feet = {leg: pos[:, ix[f]] for leg, (_, f) in SKELETON["legs"].items()}
  front = (roots["front_r"] + roots["front_l"]) / 2
  back = (roots["back_r"] + roots["back_l"]) / 2
  x = front - back
  x /= np.linalg.norm(x, axis=1, keepdims=True)
  y = (roots["front_l"] - roots["front_r"]) + (roots["back_l"] - roots["back_r"])
  y -= (y * x).sum(1, keepdims=True) * x
  y /= np.linalg.norm(y, axis=1, keepdims=True)
  R = np.stack([x, y, np.cross(x, y)], -1)
  targets = {
    leg: np.column_stack([f[:, 0], f[:, 1], np.maximum(f[:, 2], md.FOOT_RADIUS)])
    for leg, f in feet.items()
  }
  rp, rq, jq = solve_pupper((front + back) / 2, R, targets)
  motion = md.Motion(fps=FPS, root_pos=rp, root_quat=rq, joint_pos=jq)

  k = md.kinematics()
  errs = []
  for f in range(0, len(rp), 5):
    got = k.feet(rp[f], rq[f], jq[f])
    errs += [np.linalg.norm(got[leg] - targets[leg][f]) for leg in md.LEGS]
  xy = rp[:, :2]
  w = max(1, int(0.25 * FPS))  # peak speed over a quarter second, to skip jitter
  peak = (
    float(np.max(np.linalg.norm(xy[w:] - xy[:-w], axis=1)) / (w / FPS))
    if len(xy) > w
    else 0.0
  )
  feet_z = np.stack([t[:, 2] for t in targets.values()], 1) - md.FOOT_RADIUS  # (T, 4)
  stats = dict(md.check_motion(motion))
  stats.update(
    travel_m=round(float(np.linalg.norm(xy[-1] - xy[0])), 3),
    peak_speed_m_s=round(peak, 2),
    ik_foot_error_median_cm=round(100 * float(np.median(errs)), 2),
    ik_foot_error_worst_cm=round(100 * float(np.max(errs)), 2),
    # Frames where no foot touches the floor: UniMate's motion can float.
    frames_with_no_foot_down=int(np.sum(feet_z.min(1) > 0.01)),
  )
  return Fit(motion=motion, skeleton=pos, feet=targets, stats=stats)


# ----------------------------------------------------------------------------- video
def render(
  fits: list[Fit],
  labels: list[str] | None = None,
  width: int = 400,
  height: int = 320,
  skeleton: bool = True,
  follow: bool = True,
) -> np.ndarray:
  """Frames (T, height, width * len(fits), 3) of each fit side by side, at 50 fps.

  Pupper is drawn grey, the generated skeleton (``skeleton=True``) as lines:
  left side red, right side blue, spine orange. Needs a GL context: on Colab or
  a headless machine set ``MUJOCO_GL=egl`` before mujoco is imported.
  """
  k = md.kinematics()
  model = _scene(width, height)
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height, width)
  cam = mujoco.MjvCamera()
  cam.distance, cam.azimuth, cam.elevation = 0.75, 135.0, -20.0
  opt = mujoco.MjvOption()
  opt.geomgroup[3] = 0  # the collision hulls
  T = max(len(f.motion.root_pos) for f in fits)
  parents = SKELETON["parents"]
  names = SKELETON["names"]
  colors = [
    (0.9, 0.15, 0.15, 1)
    if "left" in n
    else (0.1, 0.35, 0.9, 1)
    if "right" in n
    else (1, 0.65, 0.1, 1)
    for n in names
  ]
  panels = []
  for f in fits:
    frames = np.zeros((T, height, width, 3), np.uint8)
    n = len(f.motion.root_pos)
    for t in range(T):
      i = min(t, n - 1)
      data.qpos[:3] = f.motion.root_pos[i]
      data.qpos[3:7] = f.motion.root_quat[i]
      data.qpos[k.qadr] = f.motion.joint_pos[i]
      mujoco.mj_forward(model, data)
      look = f.motion.root_pos[i] if follow else f.motion.root_pos[0]
      cam.lookat[:] = (look[0], look[1], 0.1)
      renderer.update_scene(data, cam, opt)
      if skeleton:
        scn = renderer.scene
        p = f.skeleton[i]
        for j, par in enumerate(parents):
          if par < 0 or scn.ngeom >= scn.maxgeom:
            continue
          g = scn.geoms[scn.ngeom]
          mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array(colors[j], np.float32),
          )
          mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, p[par], p[j])
          scn.ngeom += 1
      frames[t] = renderer.render()
    panels.append(frames)
  renderer.close()
  video = np.concatenate(panels, axis=2)
  if labels:
    _label(video, labels, width)
  return video


def _scene(width: int, height: int) -> mujoco.MjModel:
  """Pupper on a checkered floor under a sky, for videos (same qpos as the IK model)."""
  spec = get_spec_with_hulls()
  spec.visual.global_.offwidth = max(width, 640)
  spec.visual.global_.offheight = max(height, 480)
  spec.visual.headlight.ambient = np.full(3, 0.35, np.float32)
  spec.visual.headlight.diffuse = np.full(3, 0.6, np.float32)
  spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.55, 0.7, 0.9),
    rgb2=(0.95, 0.97, 1.0),
    width=256,
    height=256,
  )
  spec.add_texture(
    name="checker",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.82, 0.84, 0.86),
    rgb2=(0.72, 0.74, 0.77),
    width=256,
    height=256,
  )
  floor = spec.add_material(name="floor", texrepeat=(4, 4), texuniform=True)
  floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "checker"
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(20, 20, 0.1),
    material="floor",
    contype=0,
    conaffinity=0,
  )
  spec.worldbody.add_light(pos=(0, 0, 3), dir=(0, 0, -1), castshadow=True)
  return spec.compile()


def _label(video: np.ndarray, labels: list[str], width: int) -> None:
  """Write each panel's label in its top-left corner (in place)."""
  for t in range(len(video)):
    img = Image.fromarray(video[t])
    draw = ImageDraw.Draw(img)
    for p, text in enumerate(labels):
      draw.text((p * width + 8, 6), text, fill=(0, 0, 0))
    video[t] = np.asarray(img)
