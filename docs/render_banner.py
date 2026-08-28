"""Render the README banner: uv run python docs/render_banner.py"""

import numpy as np
import mujoco
from PIL import Image

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
from mjlab.tasks.pupper_gait.mdp import gait_reference

XML = str(REPO / "src/mjlab/asset_zoo/robots/pupper_v3/xmls/pupper_v3.xml")
OUT = str(REPO / "docs/pupper-banner.jpg")

W, H = 2400, 800

trot = gait_reference.build_joint_reference_table(100, "trot")  # (100, 12)

# (label, trot phase index, xy pos, yaw deg, extra pitch deg, z offset, rgba)
GRAPHITE = (0.23, 0.24, 0.26, 1)
CARDINAL = (0.55, 0.08, 0.09, 1)
SLATE = (0.32, 0.37, 0.44, 1)

# Hand-posed leap: front legs tucked, hind legs extended for take-off.
LEAP_JOINTS = np.array([
    0.25, 1.5, -2.0,    # front_r tucked
    -0.25, -1.5, 2.0,   # front_l tucked
    0.30, -0.42, -0.05, # back_r trailing
    -0.30, 0.42, 0.05,  # back_l trailing
])

ROBOTS = [
    ("walk1", 12, (-0.85, 0.12), 10, 0, 0.0, GRAPHITE),
    ("leap", None, (0.0, -0.08), 100, -20, 0.15, CARDINAL),
    ("walk2", 62, (0.85, 0.12), 190, 0, 0.0, SLATE),
]


def euler_to_quat(yaw_deg, pitch_deg, roll_deg=0.0):
    cy, sy = np.cos(np.radians(yaw_deg) / 2), np.sin(np.radians(yaw_deg) / 2)
    cp, sp = np.cos(np.radians(pitch_deg) / 2), np.sin(np.radians(pitch_deg) / 2)
    cr, sr = np.cos(np.radians(roll_deg) / 2), np.sin(np.radians(roll_deg) / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


spec = mujoco.MjSpec()
spec.modelname = "pupper_banner"

# Visual quality.
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H
spec.visual.quality.shadowsize = 8192
spec.visual.quality.offsamples = 8
spec.visual.headlight.ambient = (0.35, 0.35, 0.38)
spec.visual.headlight.diffuse = (0.25, 0.25, 0.25)
spec.visual.headlight.specular = (0.0, 0.0, 0.0)

# Sky and floor.
spec.add_texture(
    name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.92, 0.94, 0.97), rgb2=(0.75, 0.80, 0.88), width=512, height=512,
)
spec.add_texture(
    name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.87, 0.88, 0.90), rgb2=(0.83, 0.84, 0.87),
    width=512, height=512, mark=mujoco.mjtMark.mjMARK_EDGE,
    markrgb=(0.74, 0.75, 0.78),
)
mat = spec.add_material(name="floor_mat", reflectance=0.0, texrepeat=(28, 28))
mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(6, 6, 0.1),
    material="floor_mat", contype=0, conaffinity=0,
)

spec.worldbody.add_light(
    pos=(1.5, -2.5, 2.5), dir=(-0.45, 0.7, -0.8), castshadow=True,
    diffuse=(0.75, 0.75, 0.73), specular=(0.3, 0.3, 0.3),
)
spec.worldbody.add_light(
    pos=(-2.0, 1.5, 2.0), dir=(0.6, -0.4, -0.8), castshadow=False,
    diffuse=(0.28, 0.28, 0.32), specular=(0.0, 0.0, 0.0),
)

# Attach the robots.
for name, phase, xy, yaw, pitch, dz, rgba in ROBOTS:
    child = mujoco.MjSpec.from_file(XML)
    for g in child.geoms:
        if g.classname and g.classname.name == "collision":
            continue
        g.rgba = rgba
    frame = spec.worldbody.add_frame(pos=(xy[0], xy[1], 0.0))
    spec.attach(child, prefix=f"{name}_", frame=frame)

model = spec.compile()
data = mujoco.MjData(model)

for name, phase, xy, yaw, pitch, dz, rgba in ROBOTS:
    joints = LEAP_JOINTS if phase is None else trot[phase]
    jadr = model.joint(f"{name}_world_to_body").qposadr[0]
    data.qpos[jadr + 3 : jadr + 7] = euler_to_quat(yaw, pitch)
    for jname, angle in zip(
        ("leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
         "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
         "leg_back_r_1", "leg_back_r_2", "leg_back_r_3",
         "leg_back_l_1", "leg_back_l_2", "leg_back_l_3"),
        joints,
    ):
        data.qpos[model.joint(f"{name}_{jname}").qposadr[0]] = angle
    # Drop the base so the lowest foot kisses the floor, then apply dz.
    data.qpos[jadr : jadr + 3] = (xy[0], xy[1], 0.30)
    mujoco.mj_forward(model, data)
    feet = [
        data.site(f"{name}_leg_{leg}_3_foot_site").xpos[2]
        for leg in ("front_r", "front_l", "back_r", "back_l")
    ]
    foot_r = 0.019
    data.qpos[jadr + 2] = 0.30 - min(feet) + foot_r + dz
mujoco.mj_forward(model, data)

cam = mujoco.MjvCamera()
cam.lookat = (0.0, 0.0, 0.17)
cam.distance = 1.50
cam.azimuth = 90
cam.elevation = -9

renderer = mujoco.Renderer(model, height=H, width=W)
renderer.update_scene(data, camera=cam)
img = renderer.render()
Image.fromarray(img).convert("RGB").save(OUT, quality=90)
print("saved", OUT)
