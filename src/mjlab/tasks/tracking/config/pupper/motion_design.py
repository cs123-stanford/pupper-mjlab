"""Design reference motions ("tricks") for Pupper from a few keyframes.

A keyframe says where the body is and where each foot is, in the world:

    Keyframe(t=1.5, root_pos=(0.02, 0, 0.12), root_rpy=(0, -30, 0),
             feet={"front_r": (0.10, -0.09, 0.02), ...})

Inverse kinematics turns every frame into joint angles, so a foot you put on
the ground stays on the ground. Root pose + joint angles over time is a
*motion*, and one motion file serves three uses:

1. Replay on the robot as a joint-trajectory trick (``save_robot_csv``).
2. Replay in simulation, with or without physics, to see whether plain replay
   holds up (``play_motion.py``).
3. Train a tracking policy (BeyondMimic) that makes the motion physically
   feasible (``save_npz`` feeds ``Mjlab-Tracking-Flat-Pupper-v3``).

Conventions
- World frame: x forward, y left, z up; the ground is z = 0.
- ``root_pos`` is the ``base_link`` origin. At the default stand it sits about
  0.16 m above the ground.
- ``root_rpy`` is roll, pitch, yaw in degrees, applied as intrinsic
  z-y-x (yaw, then pitch, then roll). Negative pitch lifts the nose.
- Foot targets are the centre of the foot ball (radius 0.02 m), so a foot
  resting on the ground has z = FOOT_RADIUS.
- Joint order is ``JOINT_NAMES`` (front_r, front_l, back_r, back_l; each
  abduction, hip, knee), the same order the robot uses.

Run ``python -m mjlab.tasks.tracking.config.pupper.tricks.sit`` for a
worked example.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import DEFAULT_POSE, JOINT_NAMES
from mjlab.tasks.tracking.config.pupper.env_cfgs import HULL_BODIES, get_spec_with_hulls

LEGS = ("front_r", "front_l", "back_r", "back_l")
FOOT_RADIUS = 0.01995
STAND_HEIGHT = 0.161  # base_link height at the default pose with feet on the ground

# Points on the torso that typically touch the ground, in base_link frame: the
# rear and front bottom edges of the torso mesh (its extent is x -0.108..0.151,
# y -0.061..0.061, z -0.032..0.128).
BUTT = (-0.108, 0.0, -0.032)
CHEST = (0.151, 0.0, -0.032)


# ----------------------------------------------------------------------------- rotations
def rpy_to_quat(rpy_deg) -> np.ndarray:
  """(roll, pitch, yaw) in degrees -> quaternion (w, x, y, z)."""
  r, p, y = (math.radians(a) for a in rpy_deg)
  cr, sr, cp, sp, cy, sy = (
    math.cos(r / 2),
    math.sin(r / 2),
    math.cos(p / 2),
    math.sin(p / 2),
    math.cos(y / 2),
    math.sin(y / 2),
  )
  return np.array(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ]
  )


def quat_to_mat(q) -> np.ndarray:
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
  return m.reshape(3, 3)


# ----------------------------------------------------------------------------- model
class PupperKinematics:
  """Forward and inverse kinematics on the Pupper MuJoCo model."""

  def __init__(self) -> None:
    self.model = get_spec_with_hulls().compile()
    self.data = mujoco.MjData(self.model)
    m = self.model
    self.qadr = np.array(
      [
        m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in JOINT_NAMES
      ]
    )
    self.dadr = np.array(
      [
        m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in JOINT_NAMES
      ]
    )
    self.lower = np.array(
      [
        m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
        for n in JOINT_NAMES
      ]
    )
    self.upper = np.array(
      [
        m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
        for n in JOINT_NAMES
      ]
    )
    self.foot_site = {
      leg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"leg_{leg}_3_foot_site")
      for leg in LEGS
    }
    self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    # Bodies in the order the tracking task indexes them (every body but the world).
    self.body_names = [
      mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)
    ]
    # Collision-hull vertices per body (subsampled), for ground-clearance checks.
    self.hull_points = {}
    for b in HULL_BODIES:
      g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{b}_hull")
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]][
        ::7
      ]
      R = np.zeros(9)
      mujoco.mju_quat2Mat(R, m.geom_quat[g])
      self.hull_points[g] = v @ R.reshape(3, 3).T + m.geom_pos[g]  # in the body frame

  def set_state(self, root_pos, root_quat, joints) -> None:
    d = self.data
    d.qpos[:3] = root_pos
    d.qpos[3:7] = root_quat
    d.qpos[self.qadr] = joints
    mujoco.mj_kinematics(self.model, d)
    mujoco.mj_comPos(self.model, d)  # needed by mj_jacSite

  def feet(self, root_pos, root_quat, joints) -> dict[str, np.ndarray]:
    self.set_state(root_pos, root_quat, joints)
    return {leg: self.data.site_xpos[s].copy() for leg, s in self.foot_site.items()}

  def solve_leg(
    self, leg, target, root_pos, root_quat, joints, iters=60
  ) -> tuple[np.ndarray, float]:
    """Move one leg's three joints so its foot reaches ``target`` (damped least squares)."""
    idx = slice(3 * LEGS.index(leg), 3 * LEGS.index(leg) + 3)
    q = np.array(joints, dtype=float)
    jacp = np.zeros((3, self.model.nv))
    err = 1.0
    for _ in range(iters):
      self.set_state(root_pos, root_quat, q)
      err_vec = np.asarray(target) - self.data.site_xpos[self.foot_site[leg]]
      err = float(np.linalg.norm(err_vec))
      if err < 1e-5:
        break
      mujoco.mj_jacSite(self.model, self.data, jacp, None, self.foot_site[leg])
      J = jacp[:, self.dadr[idx]]
      dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err_vec)
      q[idx] = np.clip(q[idx] + dq, self.lower[idx], self.upper[idx])
    return q, err

  def lowest_body_point(self) -> float:
    """Lowest point of any collision hull for the state set last (world z)."""
    d = self.data
    low = 1.0
    for g, pts in self.hull_points.items():
      b = self.model.geom_bodyid[g]
      low = min(low, float((d.xpos[b] + pts @ d.xmat[b].reshape(3, 3).T)[:, 2].min()))
    return low

  def body_point_world(self, root_pos, root_quat, point_body) -> np.ndarray:
    return np.asarray(root_pos) + quat_to_mat(root_quat) @ np.asarray(point_body)


KIN = None


def kinematics() -> PupperKinematics:
  global KIN
  if KIN is None:
    KIN = PupperKinematics()
  return KIN


# ----------------------------------------------------------------------------- keyframes
@dataclasses.dataclass
class Keyframe:
  """The robot's pose at time ``t`` (seconds).

  ``feet`` maps a leg name to a world foot position; legs left out keep the
  joint angles from ``joints`` (or the default stand). ``joints`` optionally
  sets angles directly, in JOINT_NAMES order, for legs that aren't touching
  anything (a raised paw, say).
  """

  t: float
  root_pos: tuple[float, float, float]
  root_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
  feet: dict[str, tuple[float, float, float]] = dataclasses.field(default_factory=dict)
  joints: tuple[float, ...] | None = None


def stand_feet(
  root_pos=(0.0, 0.0, STAND_HEIGHT), root_rpy=(0, 0, 0)
) -> dict[str, tuple]:
  """Foot positions of the default stand, placed on the ground under ``root_pos``."""
  k = kinematics()
  feet = k.feet(
    np.array([0.0, 0.0, STAND_HEIGHT]), np.array([1.0, 0, 0, 0]), np.array(DEFAULT_POSE)
  )
  q = rpy_to_quat((0, 0, root_rpy[2]))
  R = quat_to_mat(q)
  out = {}
  for leg, p in feet.items():
    xy = R @ np.array([p[0], p[1], 0.0])
    out[leg] = (root_pos[0] + xy[0], root_pos[1] + xy[1], FOOT_RADIUS)
  return out


def stand(t: float, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> Keyframe:
  """The default standing pose at time ``t``."""
  root = (x, y, STAND_HEIGHT)
  return Keyframe(
    t=t, root_pos=root, root_rpy=(0.0, 0.0, yaw), feet=stand_feet(root, (0, 0, yaw))
  )


def root_with_point_on_ground(
  point_body, root_rpy, x=None, y=0.0, ground_z=0.0
) -> tuple[float, float, float]:
  """Root position that puts a body point (e.g. BUTT) on the ground at the given pitch/roll.

  If ``x`` is given, the *point* lands at that world x; otherwise the root stays at x = 0.
  """
  R = quat_to_mat(rpy_to_quat(root_rpy))
  p = R @ np.asarray(point_body)
  rx = (x - p[0]) if x is not None else 0.0
  return (float(rx), float(y - p[1]), float(ground_z - p[2]))


def settle_on_ground(kf: Keyframe, clearance: float = 0.0, iters: int = 30) -> Keyframe:
  """Raise or lower a keyframe's root so its lowest collision hull just touches the ground.

  Use it for poses that rest on the body or legs (sitting, lying down): pinned
  feet stay where they are and the legs are re-solved at every height tried.
  Returns a copy of ``kf`` with the adjusted root height.
  """
  k = kinematics()
  rq = rpy_to_quat(kf.root_rpy)

  def lowest(dz: float) -> float:
    root = np.array(kf.root_pos, dtype=float) + np.array([0.0, 0.0, dz])
    q = np.array(kf.joints if kf.joints is not None else DEFAULT_POSE, dtype=float)
    for leg, target in kf.feet.items():
      q, _ = k.solve_leg(leg, target, root, rq, q, iters=200)
    k.set_state(root, rq, q)
    return k.lowest_body_point() - clearance

  lo, hi = -0.05, 0.05  # search +-5 cm around the given height
  for _ in range(iters):
    mid = 0.5 * (lo + hi)
    if lowest(mid) < 0.0:
      lo = mid
    else:
      hi = mid
  root = tuple(float(v) for v in np.array(kf.root_pos) + np.array([0.0, 0.0, hi]))
  return dataclasses.replace(kf, root_pos=root)


# ----------------------------------------------------------------------------- motion
@dataclasses.dataclass
class Motion:
  fps: float
  root_pos: np.ndarray  # (T, 3)
  root_quat: np.ndarray  # (T, 4) w, x, y, z
  joint_pos: np.ndarray  # (T, 12) JOINT_NAMES order
  report: dict = dataclasses.field(default_factory=dict)
  # Optional clip table (see concat_clips): name, start/end frame (end exclusive),
  # loop, and the clip that plays next by default.
  clips: list[dict] = dataclasses.field(default_factory=list)

  @property
  def duration(self) -> float:
    return (len(self.root_pos) - 1) / self.fps

  # --- files -----------------------------------------------------------------
  def save_csv(self, path) -> Path:
    """BeyondMimic motion CSV: no header, rows x, y, z, qx, qy, qz, qw, 12 joints.

    Written at ``self.fps``; pass ``--input-fps`` with that value when converting.
    """
    q_xyzw = self.root_quat[:, [1, 2, 3, 0]]
    rows = np.concatenate([self.root_pos, q_xyzw, self.joint_pos], axis=1)
    path = Path(path)
    np.savetxt(path, rows, delimiter=",", fmt="%.6f")
    return path

  def save_npz(self, path) -> Path:
    """Tracking-task motion file: joint and body states at ``self.fps`` (default 50 Hz)."""
    k = kinematics()
    m, d = k.model, k.data
    dt = 1.0 / self.fps
    T = len(self.root_pos)
    nb = m.nbody - 1
    body_pos = np.zeros((T, nb, 3))
    body_quat = np.zeros((T, nb, 4))
    qpos = np.zeros((T, m.nq))
    for i in range(T):
      k.set_state(self.root_pos[i], self.root_quat[i], self.joint_pos[i])
      qpos[i] = d.qpos
      body_pos[i] = d.xpos[1:]
      body_quat[i] = d.xquat[1:]
    # Generalized velocities by finite differences, then body velocities in the world frame.
    qvel = np.zeros((T, m.nv))
    for i in range(T):
      a, b = max(i - 1, 0), min(i + 1, T - 1)
      mujoco.mj_differentiatePos(m, qvel[i], (b - a) * dt, qpos[a], qpos[b])
    body_lin = np.zeros((T, nb, 3))
    body_ang = np.zeros((T, nb, 3))
    vel6 = np.zeros(6)
    for i in range(T):
      d.qpos[:] = qpos[i]
      d.qvel[:] = qvel[i]
      mujoco.mj_forward(m, d)
      for bi in range(1, m.nbody):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bi, vel6, 0)
        body_ang[i, bi - 1] = vel6[:3]
        body_lin[i, bi - 1] = vel6[3:]
    joint_vel = qvel[:, k.dadr]
    path = Path(path)
    np.savez(
      path,
      fps=np.array([self.fps]),
      joint_pos=self.joint_pos.astype(np.float32),
      joint_vel=joint_vel.astype(np.float32),
      body_pos_w=body_pos.astype(np.float32),
      body_quat_w=body_quat.astype(np.float32),
      body_lin_vel_w=body_lin.astype(np.float32),
      body_ang_vel_w=body_ang.astype(np.float32),
      body_names=np.array(k.body_names),
      **clip_arrays(self.clips),
    )
    return path

  def save_robot_csv(self, path) -> Path:
    """Joint-trajectory trick for the robot's animation player (absolute radians).

    Same format as the Pupper monorepo's trick recordings: a header of
    timestamp_ns, timestamp_sec and the 12 joint names, one row per frame.
    """
    path = Path(path)
    t = np.arange(len(self.joint_pos)) / self.fps
    header = "timestamp_ns,timestamp_sec," + ",".join(JOINT_NAMES)
    rows = np.concatenate([(t * 1e9)[:, None], t[:, None], self.joint_pos], axis=1)
    np.savetxt(
      path,
      rows,
      delimiter=",",
      header=header,
      comments="",
      fmt=["%d", "%.6f"] + ["%.6f"] * 12,
    )
    return path


def _min_jerk(s: np.ndarray) -> np.ndarray:
  return 10 * s**3 - 15 * s**4 + 6 * s**5


def build_motion(
  keyframes: list[Keyframe], fps: float = 50.0, keep_above_ground: bool = True
) -> Motion:
  """Interpolate keyframes smoothly and solve every frame with inverse kinematics.

  Root position, orientation and foot positions move along minimum-jerk curves
  between consecutive keyframes (zero velocity at each keyframe). Two keyframes
  with the same pose make a hold.

  With ``keep_above_ground`` (the default), any frame where a collision hull
  would go below the ground has its root raised just enough, with the pinned
  feet re-solved, so the motion never passes through the floor between keyframes.
  """
  k = kinematics()
  kfs = sorted(keyframes, key=lambda kf: kf.t)
  if kfs[0].t != 0:
    raise ValueError("the first keyframe must be at t = 0")
  default = np.array(DEFAULT_POSE, dtype=float)
  # Fill each keyframe's foot targets: explicit ones, else wherever its joints put them.
  resolved = []
  for kf in kfs:
    q = np.array(kf.joints if kf.joints is not None else default, dtype=float)
    rq = rpy_to_quat(kf.root_rpy)
    feet = k.feet(np.array(kf.root_pos), rq, q)
    feet.update({leg: np.array(p, dtype=float) for leg, p in kf.feet.items()})
    resolved.append(
      (
        np.array(kf.root_pos, dtype=float),
        np.array(kf.root_rpy, dtype=float),
        feet,
        q,
        set(kf.feet),
      )
    )

  # Solve each keyframe on its own first, seeded from the default stand (then the
  # previous keyframe), so every keyframe lands on a sensible leg configuration.
  key_q = []
  for rp, rpy, feet, q0, pinned in resolved:
    best = None
    for seed in [default, key_q[-1]] if key_q else [default]:
      q = q0.copy()
      for leg in pinned:
        sl = slice(3 * LEGS.index(leg), 3 * LEGS.index(leg) + 3)
        q[sl] = seed[sl]
      err = 0.0
      for leg in pinned:
        q, e = k.solve_leg(leg, feet[leg], rp, rpy_to_quat(rpy), q, iters=200)
        err = max(err, e)
      if best is None or err < best[1] - 1e-4:
        best = (q, err)
    assert best is not None
    key_q.append(best[0])

  n = int(round(kfs[-1].t * fps)) + 1
  times = np.arange(n) / fps
  root_pos = np.zeros((n, 3))
  root_quat = np.zeros((n, 4))
  joints = np.zeros((n, 12))
  q = key_q[0].copy()
  worst_err = 0.0
  for i, t in enumerate(times):
    j = min(np.searchsorted([kf.t for kf in kfs], t, side="right") - 1, len(kfs) - 2)
    a, b = resolved[j], resolved[j + 1]
    span = kfs[j + 1].t - kfs[j].t
    s = float(_min_jerk(np.clip((t - kfs[j].t) / span, 0.0, 1.0))) if span > 0 else 1.0
    rp = a[0] + s * (b[0] - a[0])
    rq = rpy_to_quat(a[1] + s * (b[1] - a[1]))
    q_blend = key_q[j] + s * (key_q[j + 1] - key_q[j])
    q_new = q_blend.copy()
    for leg in LEGS:
      sl = slice(3 * LEGS.index(leg), 3 * LEGS.index(leg) + 3)
      if leg not in a[4] and leg not in b[4]:
        continue  # free leg: keep the blended joint angles
      target = a[2][leg] + s * (b[2][leg] - a[2][leg])
      # Two seeds: continue from the last frame, or start from the blend of the
      # neighbouring keyframe solutions. Keep whichever reaches the target;
      # on a tie prefer the one closer to the last frame (smooth motion).
      candidates = []
      for seed in (q, q_blend):
        trial = q_new.copy()
        trial[sl] = seed[sl]
        trial, e = k.solve_leg(leg, target, rp, rq, trial)
        jump = float(np.abs(trial[sl] - q[sl]).max()) if i else 0.0
        candidates.append((round(e, 4), jump, trial[sl]))
      e, _, leg_q = min(candidates, key=lambda c: (c[0], c[1]))
      q_new[sl] = leg_q
      worst_err = max(worst_err, e)
    q = q_new
    if keep_above_ground:
      pinned = [leg for leg in LEGS if leg in a[4] or leg in b[4]]
      targets = {leg: a[2][leg] + s * (b[2][leg] - a[2][leg]) for leg in pinned}
      for _ in range(4):
        k.set_state(rp, rq, q)
        low = k.lowest_body_point()
        if low >= -1e-4:
          break
        rp = rp + np.array([0.0, 0.0, -low])
        for leg in pinned:
          q, _ = k.solve_leg(leg, targets[leg], rp, rq, q)
    root_pos[i], root_quat[i], joints[i] = rp, rq, q

  motion = Motion(fps=fps, root_pos=root_pos, root_quat=root_quat, joint_pos=joints)
  motion.report = check_motion(motion)
  motion.report["max_ik_error_m"] = round(worst_err, 4)
  return motion


@dataclasses.dataclass
class Clip:
  """One clip of a trick. ``next`` names the clip that plays when this one ends
  (None: hold the last frame); a looping clip repeats until another is requested."""

  name: str
  motion: Motion
  loop: bool = False
  next: str | None = None


def concat_clips(clips: list[Clip], order: list[str] | None = None) -> Motion:
  """Join clips into one continuous motion for training, with a clip table.

  Every clip must start and end at poses that match the clips it can follow and
  precede (usually one shared pose, like "sitting"), so the robot can jump from
  the end of any clip to the start of the next without a discontinuity. The
  tracking policy trains on the whole joined motion; on the robot the controller
  plays one clip at a time and switches clips at clip boundaries.

  ``order`` is the training sequence (names may repeat, e.g. an idle clip between
  every action); by default each clip once, in the given order. The clip table
  points at each clip's first occurrence.
  """
  by_name = {c.name: c for c in clips}
  order = order or [c.name for c in clips]
  fps = clips[0].motion.fps
  parts, table, frame = [], {}, 0
  for i, name in enumerate(order):
    m = by_name[name].motion
    if m.fps != fps:
      raise ValueError(f"clip {name} is at {m.fps} fps, expected {fps}")
    if i > 0:
      prev = by_name[order[i - 1]].motion
      gap = np.abs(prev.joint_pos[-1] - m.joint_pos[0]).max()
      if gap > 0.05:
        raise ValueError(
          f"{order[i - 1]} ends {gap:.2f} rad away from where {name} starts"
        )
    # Frames shared at a boundary are kept once: drop this clip's first frame.
    skip = 1 if i > 0 else 0
    start = frame - skip
    n = len(m.root_pos)
    if name not in table:
      c = by_name[name]
      table[name] = {
        "name": name,
        "start": max(start, 0),
        "end": max(start, 0) + n,
        "loop": c.loop,
        "next": c.next,
      }
    parts.append((m.root_pos[skip:], m.root_quat[skip:], m.joint_pos[skip:]))
    frame += n - skip
  motion = Motion(
    fps=fps,
    root_pos=np.concatenate([p[0] for p in parts]),
    root_quat=np.concatenate([p[1] for p in parts]),
    joint_pos=np.concatenate([p[2] for p in parts]),
    clips=[table[c.name] for c in clips],
  )
  motion.report = check_motion(motion)
  return motion


def clip_arrays(clips: list[dict]) -> dict:
  """The clip table as npz arrays (empty when there are no clips)."""
  if not clips:
    return {}
  return {
    "clip_names": np.array([c["name"] for c in clips]),
    "clip_starts": np.array([c["start"] for c in clips]),
    "clip_ends": np.array([c["end"] for c in clips]),
    "clip_loops": np.array([bool(c["loop"]) for c in clips]),
    "clip_nexts": np.array([c["next"] or "" for c in clips]),
  }


def read_clips(npz_path) -> list[dict]:
  """The clip table stored in a motion npz, or [] for a single-clip motion."""
  data = np.load(npz_path)
  if "clip_names" not in data:
    return []
  return [
    {
      "name": str(n),
      "start": int(s),
      "end": int(e),
      "loop": bool(lp),
      "next": str(nx) or None,
    }
    for n, s, e, lp, nx in zip(
      data["clip_names"],
      data["clip_starts"],
      data["clip_ends"],
      data["clip_loops"],
      data["clip_nexts"],
      strict=True,
    )
  ]


def check_motion(motion: Motion) -> dict:
  """Sanity checks a student (or their coding agent) should read before training."""
  k = kinematics()
  lowest_foot = 1.0
  lowest_body = 1.0
  for i in range(len(motion.root_pos)):
    feet = k.feet(motion.root_pos[i], motion.root_quat[i], motion.joint_pos[i])
    lowest_foot = min(lowest_foot, min(p[2] for p in feet.values()) - FOOT_RADIUS)
    lowest_body = min(lowest_body, k.lowest_body_point())
  at_limit = np.mean(
    (motion.joint_pos <= k.lower + 1e-3) | (motion.joint_pos >= k.upper - 1e-3)
  )
  jvel = np.abs(np.diff(motion.joint_pos, axis=0)) * motion.fps
  return {
    "duration_s": round(motion.duration, 2),
    "frames": len(motion.root_pos),
    "lowest_foot_bottom_m": round(
      lowest_foot, 4
    ),  # < 0 means a foot goes into the ground
    "lowest_body_point_m": round(
      lowest_body, 4
    ),  # torso/leg hulls; < 0 means into the ground
    "fraction_of_joint_samples_at_limit": round(float(at_limit), 3),
    "max_joint_speed_rad_s": round(
      float(jvel.max()), 2
    ),  # the motors top out around 20 rad/s
  }
