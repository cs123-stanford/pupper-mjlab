import os
import sys

# Default to EGL for GPU-accelerated offscreen rendering on Linux. Must be set
# before any mujoco import: mujoco's gl_context module captures MUJOCO_GL once
# at load time. An explicit MUJOCO_GL in the environment always wins. When
# unset, probe EGL first and fall back to osmesa (CPU rendering) or disabled
# on machines without an EGL stack, so headless training still imports —
# rendering there is warp-side and needs no GL; only videos/native viewing do.
# Linux-only because mujoco's gl_context rejects "egl" on macOS/Windows and
# raises at import. On those platforms we leave MUJOCO_GL alone so mujoco
# defaults to GLFW.
# (Everything lives inside the `if` so no statements precede the imports
# below other than allowed conditionals — keeps E402 happy.)
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:

  def _egl_usable() -> bool:
    """Whether a headless EGL context could plausibly be created.

    With MUJOCO_GL=egl, ``import mujoco`` eagerly initializes an EGL
    display and DIES AT IMPORT on machines without an EGL userland (e.g.
    CUDA compute-only cluster containers, which ship libcuda but not
    libEGL/libglvnd) — even though training never renders through GL
    (camera sensors raytrace via mujoco-warp). Probe with ctypes: load
    libEGL and enumerate EGL devices, mirroring what mujoco.egl requires
    (the EXT_platform_device path).
    """
    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("EGL") or "libEGL.so.1"
    try:
      egl = ctypes.CDLL(name)
      egl.eglGetProcAddress.restype = ctypes.c_void_p
      egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
      addr = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
      if not addr:
        return False
      proto = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
      )
      devices = (ctypes.c_void_p * 8)()
      num = ctypes.c_int(0)
      if not proto(addr)(8, devices, ctypes.byref(num)):
        return False
      return num.value > 0
    except Exception:
      return False

  if _egl_usable():
    os.environ["MUJOCO_GL"] = "egl"
  else:
    import ctypes.util as _ctypes_util

    _fallback = "osmesa" if _ctypes_util.find_library("OSMesa") else "disabled"
    os.environ["MUJOCO_GL"] = _fallback
    print(
      f"[mjlab] No usable EGL device found; defaulting MUJOCO_GL={_fallback}. "
      "Training is unaffected (cameras raytrace via warp), but video "
      "recording and native viewing need EGL (install libEGL / run the "
      "container with NVIDIA graphics capability) or set MUJOCO_GL "
      "explicitly.",
      file=sys.stderr,
    )

import traceback
from importlib.metadata import entry_points
from pathlib import Path

import tyro
import warp as wp

MJLAB_SRC_PATH: Path = Path(__file__).parent

TYRO_FLAGS = (
  # Don't let users switch between types in unions. This produces a simpler CLI
  # with flatter helptext, at the cost of some flexibility. Type changes can
  # just be done in code.
  tyro.conf.AvoidSubcommands,
  # Disable automatic flag conversion (e.g., use `--flag False` instead of
  # `--no-flag` for booleans).
  tyro.conf.FlagConversionOff,
  # Use Python syntax for collections: --tuple (1,2,3) instead of --tuple 1 2 3.
  # Helps with wandb sweep compatibility: https://brentyi.github.io/tyro/wandb_sweeps/
  tyro.conf.UsePythonSyntaxForLiteralCollections,
)


def _configure_warp() -> None:
  """Configure Warp globally for mjlab."""
  wp.config.enable_backward = False

  # Keep warp verbose by default to show kernel compilation progress.
  # Override with MJLAB_WARP_QUIET=1 environment variable if needed.
  quiet = os.environ.get("MJLAB_WARP_QUIET", "0").lower() in ("1", "true", "yes")
  wp.config.quiet = quiet


def _import_registered_packages() -> None:
  """Auto-discover and import packages registered via entry points.

  Looks for packages registered under the 'mjlab.tasks' entry point group.
  Each discovered package is imported, which allows it to register custom
  environments with gymnasium.
  """
  mjlab_tasks = entry_points().select(group="mjlab.tasks")
  for entry_point in mjlab_tasks:
    try:
      entry_point.load()
    except Exception:
      print(
        f"[WARN] Failed to load task package '{entry_point.name}' ({entry_point.value}):",
        file=sys.stderr,
      )
      traceback.print_exc(file=sys.stderr)


def _configure_mediapy() -> None:
  """Point mediapy at the bundled imageio-ffmpeg binary."""
  import imageio_ffmpeg
  import mediapy

  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


_configure_warp()
_configure_mediapy()
_import_registered_packages()
