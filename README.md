# Super Camera 🎥🌡️

**Simulated infrared (IR) imagery and per-buffer render access for NVIDIA Isaac Sim / Omniverse.**

Super Camera wraps the `omni.replicator.core` annotator pipeline into a single
`SuperCamera` class. It removes the boilerplate of wiring up render products and
annotators by hand, and turns raw render buffers (distance, normals, albedo,
roughness, …) into **synthetic thermal / near-infrared images** rendered with a
real thermal-camera ("ironbow") colormap.

It ships as both an **auto-loading Isaac Sim extension** (with a GUI panel) and a
**standalone Python library** that drops straight into a robot-learning pipeline
such as **Isaac Lab**.

<p align="center">
  <img src="docs/images/hero_thermal.png" alt="Thermal IR output with ironbow colormap" width="80%">
  <br><em>Placeholder — thermal-mode IR render (ironbow palette)</em>
</p>

---

## Table of contents

- [Features](#features)
- [Gallery](#gallery)
- [Installation](#installation)
- [Quick start](#quick-start)
- [IR synthesis](#ir-synthesis)
- [Thermal colormap](#thermal-colormap)
- [Writing your own heuristic](#writing-your-own-heuristic)
- [Isaac Lab / training-pipeline integration](#isaac-lab--training-pipeline-integration)
- [GUI panel](#gui-panel)
- [Buffer reference](#buffer-reference)
- [Mock mode](#mock-mode)
- [Project structure](#project-structure)
- [Requirements](#requirements)

---

## Features

- 🌡️ **Simulated IR** — `thermal` and `active_nir` modes built purely from
  per-buffer geometry data (no external sensor model required).
- 🎨 **Thermal colormap** — output rendered with an **ironbow** palette that looks
  like a FLIR-style thermal camera (black → purple → red → orange → yellow → white).
- 🧩 **One class, every buffer** — distance, depth, normals, albedo, roughness,
  segmentation, bounding boxes, point clouds, and more behind one API.
- 🤖 **Pipeline-ready** — a no-orchestrator-step `read()` path makes it safe to use
  inside an externally-driven loop like **Isaac Lab**.
- 🖥️ **GUI panel** — auto-loads in Isaac Sim; capture, preview, and save IR frames
  with no scripting.
- 🧪 **Mock mode** — runs anywhere with just `numpy` for CI and logic iteration.

## Gallery

| RGB | Distance (jet) | Thermal IR (ironbow) | Active NIR (ironbow) |
|---|---|---|---|
| ![rgb](docs/images/rgb.png) | ![depth](docs/images/depth_jet.png) | ![thermal](docs/images/ir_thermal.png) | ![nir](docs/images/ir_active_nir.png) |

> _Placeholders — drop your own renders into `docs/images/`._

---

## Installation

```bash
git clone https://github.com/yourusername/super_camera.git
```

### As an Isaac Sim extension

In Isaac Sim go to **Window → Extensions → ⚙ → Add path** and point it at:

```
/path/to/super_camera/exts
```

Search for **Super Camera** and enable it. A **Super Camera** window appears
automatically.

### As a standalone library

Add the package to your `PYTHONPATH` (or run from the repo root with Isaac Sim's
Python):

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/super_camera/exts/super.camera"
python standalone/example.py
```

---

## Quick start

```python
from super.camera import SuperCamera, BufferType

camera = SuperCamera(
    prim_path="/World/SuperCamera",
    resolution=(1280, 720),
    buffers=[BufferType.DISTANCE_TO_OBJECT, BufferType.NORMALS],
)

camera.aim(position=(0, -5, 2), target=(0, 0, 0))   # Z-up world

# Thermal IR → float32 (H, W) in [0, 1]
ir = camera.synthesize_ir(mode="thermal")

# Colorize for display → uint8 (H, W, 3) ironbow image
from PIL import Image
Image.fromarray(SuperCamera.colorize(ir, "ironbow")).save("ir_thermal.png")
```

---

## IR synthesis

`synthesize_ir(mode, camera_pos=None, ambient_temp=293.0)` returns a **float32
`(H, W)` array in `[0, 1]`** built from per-buffer data. Required buffers are
auto-attached on first call. Background / no-hit ("miss") rays are forced to `0`.

| Mode | Required buffers | Model |
|---|---|---|
| `thermal` | `DISTANCE_TO_OBJECT`, `NORMALS` | Distance scaffold (far = bright). **This is the place to write your own heuristic** — see below. |
| `active_nir` | `DISTANCE_TO_OBJECT`, `NORMALS`, `DIFFUSE_ALBEDO`, `SPECULAR_ALBEDO`, `ROUGHNESS` | Coaxial active-NIR illuminator: PBR reflectivity + Blinn-Phong specular + inverse-square falloff. |

```python
ir = camera.synthesize_ir(mode="thermal")
ir = camera.synthesize_ir(mode="active_nir", camera_pos=np.array([0, 0, 5]))
```

The output is already normalized to `[0, 1]` (divided by its own max) and has **no
noise** — display or save it directly. Keep the raw float map for training; only
colorize when you want to look at it.

## Thermal colormap

```python
rgb = SuperCamera.colorize(ir, colormap="ironbow")   # default — thermal-camera look
rgb = SuperCamera.colorize(ir, colormap="grayscale")
rgb = SuperCamera.colorize(ir, colormap="jet")
```

`colorize()` maps any `[0, 1]` float `(H, W)` map to a uint8 `(H, W, 3)` image. The
same colormaps are available on the depth path: `synthesize(colormap="ironbow")`.

The **ironbow** palette is defined by a small table of control stops in
[`super_camera.py`](exts/super.camera/super/camera/super_camera.py) (`_IRONBOW_STOPS`)
— edit the stops to retune the look, or add a new branch in `colorize()` for
another palette.

## Writing your own heuristic

`thermal` mode is intentionally a thin scaffold: the base on which you layer your
own per-pixel IR model. The hook is the `else` branch of `_compute_ir()` in
[`super_camera.py`](exts/super.camera/super/camera/super_camera.py), which carries
a full inline comment listing **every buffer you can access at that point**, its
shape and dtype, and how to attach the optional ones.

You only need to produce a non-negative `(H, W)` `ir_intensity` array — the shared
tail masks miss rays and normalizes to `[0, 1]` for you:

```python
# inside the `else:` (thermal) branch of _compute_ir — replace this one line:
ir_intensity = np.where(background, 0.0, bufs["DISTANCE_TO_OBJECT"])

# e.g. emissive-driven heat (add BufferType.EMISSIVE to _THERMAL_BUFFERS first):
# ir_intensity = np.mean(bufs["EMISSIVE"][:, :, :3], axis=2)
```

To make more buffers available to your heuristic, add them to `_THERMAL_BUFFERS`
at the top of `super_camera.py`, or call `camera.add_buffer(BufferType.X)` before
synthesizing.

## Isaac Lab / training-pipeline integration

Isaac Lab already advances the renderer each step via `world.step(render=True)`.
Inside such a loop, **do not** let Super Camera call `rep.orchestrator.step()` —
use the no-step read path instead:

```python
camera = SuperCamera(
    prim_path="/World/SuperCamera",
    resolution=(640, 480),
    buffers=[BufferType.DISTANCE_TO_OBJECT, BufferType.NORMALS],
)
camera.aim(position=(3, 0, 2), target=(0, 0, 0.5))

for step in range(num_steps):
    sim.step(render=True)                              # Isaac Lab renders
    ir = camera.synthesize_ir_from_render("thermal")   # reads the rendered frame, no extra step
    # ...feed `ir` into an observation / reward term...
```

| Method | Steps the orchestrator? | Use from |
|---|---|---|
| `synthesize_ir()` / `capture()` | ✅ yes (sync) | `standalone/example.py` |
| `synthesize_ir_async()` / `capture_async()` | ✅ yes (async) | the extension GUI / Kit event loop |
| `synthesize_ir_from_render()` / `read()` | ❌ no | **Isaac Lab & any externally-driven loop** |

A complete runnable example lives in
[`standalone/isaaclab_integration.py`](standalone/isaaclab_integration.py).

Super Camera attaches its **own** standalone render product to the camera prim and
never touches the active viewport, so it coexists with Isaac Lab's own sensors.

## GUI panel

When the extension loads, a **Super Camera** window appears automatically.

<p align="center">
  <img src="docs/images/gui_panel.png" alt="Super Camera GUI panel" width="45%">
  <br><em>Placeholder — extension GUI panel</em>
</p>

| Section | Controls |
|---|---|
| Camera Setup | Prim Path, Width × Height, **Create Camera**, **Open Viewport** |
| IR Mode | Mode dropdown (Thermal LWIR / Active NIR); relevant fields enabled |
| IR Preview | Live ironbow thumbnail updated after each capture |
| Output | Save Path for the PNG |
| Buttons | **Capture IR Frame**, **Reset Camera** |

Captured frames are colorized with the ironbow palette and saved as RGB PNG
(falls back to PPM if Pillow is unavailable).

## Buffer reference

Add a buffer at construction (`buffers=[...]`) or at runtime
(`camera.add_buffer(BufferType.X)`).

### Pixel buffers — `get_data()` returns `np.ndarray`

| BufferType | Annotator | Shape | dtype |
|---|---|---|---|
| `DISTANCE_TO_OBJECT` | `distance_to_camera` | `(H,W)` | float32 |
| `DEPTH` | `distance_to_image_plane` | `(H,W)` | float32 |
| `NORMALS` | `normals` | `(H,W,4)` | float32 |
| `RGB` | `rgb` | `(H,W,4)` | uint8 |
| `DIFFUSE_ALBEDO` | `diffuse_albedo` | `(H,W,4)` | float32 |
| `SPECULAR_ALBEDO` | `specular_albedo` | `(H,W,4)` | float32 |
| `ROUGHNESS` | `roughness` | `(H,W)` | float32 |
| `EMISSIVE` | `emissive` | `(H,W,4)` | float32 |
| `MOTION_VECTORS` | `motion_vectors` | `(H,W,4)` | float32 |

### Structured buffers — `get_data()` returns `dict`

| BufferType | Annotator | Notes |
|---|---|---|
| `SEMANTIC` | `semantic_segmentation` | `(H,W)` uint32 + `idToLabels` |
| `INSTANCE` | `instance_segmentation` | hierarchical instance ids |
| `INSTANCE_ID` | `instance_id_segmentation` | per-leaf-prim ids |
| `OCCLUSION` | `occlusion` | per-instance occlusion ratio |
| `BBOX_2D_TIGHT` | `bounding_box_2d_tight` | tight 2-D boxes |
| `BBOX_2D_LOOSE` | `bounding_box_2d_loose` | loose 2-D boxes |
| `BBOX_3D` | `bounding_box_3d` | 3-D boxes + world pose |
| `CAMERA_PARAMS` | `camera_params` | intrinsics / extrinsics |
| `POINTCLOUD` | `pointcloud` | `(N,3)` world-space points |
| `SKELETON` | `skeleton_data` | joint positions |

## Mock mode

No Omniverse install needed — every buffer returns zero-filled arrays of the right
shape. Useful for CI and logic iteration. Auto-enabled if
`omni.replicator.core` can't be imported.

```python
camera = SuperCamera(mock=True, buffers=[BufferType.DISTANCE_TO_OBJECT])
ir = camera.synthesize_ir(mode="thermal")
rgb = SuperCamera.colorize(ir, "ironbow")
```

```bash
python standalone/example_mock.py
```

## Project structure

```
super_camera/
├── README.md
├── CLAUDE.md                           ← development notes
├── buffers.py                          ← root copy (kept identical to ext)
├── super_camera.py                     ← root copy (kept identical to ext)
├── exts/
│   └── super.camera/
│       ├── config/extension.toml
│       └── super/camera/
│           ├── __init__.py             ← re-exports SuperCameraExtension
│           ├── buffers.py              ← BufferType / BufferData
│           ├── super_camera.py         ← SuperCamera class + colormaps
│           └── extension.py            ← GUI + Omniverse lifecycle
└── standalone/
    ├── example.py                      ← full Isaac Sim example
    ├── example_mock.py                 ← numpy-only, no Isaac Sim
    └── isaaclab_integration.py         ← Isaac Lab sim-loop example
```

## Requirements

- NVIDIA Isaac Sim 4.x / Omniverse Kit 106.x (for live capture)
- Python 3.10+
- `numpy` (and `Pillow` for PNG output)

Mock mode only needs `numpy`.
