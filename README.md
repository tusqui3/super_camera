# Super Camera 🎥🌡️

**Simulated infrared (IR) imagery and per-buffer render access for NVIDIA Isaac Sim / Omniverse.**

Super Camera wraps the `omni.replicator.core` annotator pipeline into a single
`SuperCamera` class. It removes the boilerplate of wiring up render products and
annotators by hand, and turns raw render buffers (distance, normals, albedo,
roughness, emissive, motion, …) into **synthetic multispectral IR images** —
five physically-motivated spectral bands from the visible through long-wave
infrared — rendered with a real thermal-camera ("ironbow") colormap.

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
- [Spectral bands](#spectral-bands)
- [IR synthesis](#ir-synthesis)
- [Thermal colormap](#thermal-colormap)
- [Writing your own band model](#writing-your-own-band-model)
- [Isaac Lab / training-pipeline integration](#isaac-lab--training-pipeline-integration)
- [GUI panel](#gui-panel)
- [Buffer reference](#buffer-reference)
- [Mock mode](#mock-mode)
- [Project structure](#project-structure)
- [Requirements](#requirements)

---

## Features

- 🌈 **Five spectral bands** — `VIS`, `NIR_ACTIVE`, `SWIR_ACTIVE`, `MWIR`, `LWIR`,
  each a physically-motivated model built purely from per-buffer render data (no
  external sensor model required). Reflective bands model active/ambient
  illumination return; emissive bands model thermal emission.
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

| VIS (reflective) | NIR_ACTIVE (reflective) | SWIR_ACTIVE (reflective) | MWIR (emissive) | LWIR (emissive) |
|---|---|---|---|---|
| ![vis](docs/images/ir_VIS.png) | ![nir](docs/images/ir_NIR_ACTIVE.png) | ![swir](docs/images/ir_SWIR_ACTIVE.png) | ![mwir](docs/images/ir_MWIR.png) | ![lwir](docs/images/ir_LWIR.png) |

> _Placeholders — drop your own renders into `docs/images/`. All five are produced
> by [`standalone/example.py`](standalone/example.py)._

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

# Long-wave thermal IR → float32 (H, W) in [0, 1]
ir = camera.synthesize_ir(mode="LWIR")

# Colorize for display → uint8 (H, W, 3) ironbow image
from PIL import Image
Image.fromarray(SuperCamera.colorize(ir, "ironbow")).save("ir_LWIR.png")
```

---

## Spectral bands

Super Camera models **five spectral bands**, each identified by its wavelength
range and classified as primarily **reflective** (the sensor sees illumination
bouncing off surfaces) or **emissive** (the sensor sees thermal radiation the
surface itself emits). The canonical definitions live in one place —
`SPECTRAL_BANDS` in [`buffers.py`](exts/super.camera/super/camera/buffers.py) —
and are the single source of truth read by the synthesis dispatch, the GUI, and
this table.

| Band | Wavelength | Class | Buffers used | Synthesis heuristic |
|---|---|---|---|---|
| **`VIS`** | 400–700 nm | reflective | distance, normals, diffuse, specular, roughness | Passive ambient reflectance: colour-aware diffuse `luma(diffuse)·(N·V)` + specular sheen `specular·(N·V)^(1/roughness)`. **No** distance falloff (no active source). |
| **`NIR_ACTIVE`** | 700–1000 nm | reflective | distance, normals, diffuse, specular, roughness | Coaxial active illuminator (light = camera, `H = V`). Diffuse + **boosted** specular (`×2`) so smooth surfaces glint, all divided by **distance²** (inverse-square falloff). |
| **`SWIR_ACTIVE`** | 1000–2500 nm | reflective | distance, normals, diffuse, specular, roughness | Like NIR but **colour-blind**: flat grayscale reflectance (channel mean, not luma), diffuse base modulated by `(1 − 0.5·roughness)`, even **stronger** specular (`×3`), inverse-square falloff. Emphasis on material over colour. |
| **`MWIR`** | 3000–5000 nm | emissive | distance, normals, diffuse, specular, roughness, emissive, motion | Thermal emission `ε · T⁴`. Emissivity `ε` inferred from material (rough/diffuse → high, smooth/specular → low); temperature proxy = ambient + **strongly weighted** emissive + motion heat. The `T⁴` power **biases hard toward hot objects**. |
| **`LWIR`** | 8000–14000 nm | emissive | distance, normals, diffuse, specular, roughness, emissive, motion | Ambient-temperature emission `ε · T · geom`. Near-**linear** in temperature (whole scene visible), emissive/motion add modest warmth, geometry only a weak influence, **no** distance falloff (emitted, not illuminated). |

Introspect them in code:

```python
from super.camera import SPECTRAL_BANDS
b = SPECTRAL_BANDS["MWIR"]
b.name, b.wavelength_min_nm, b.wavelength_max_nm, b.reflective_vs_emissive
# ('MWIR', 3000.0, 5000.0, 'emissive')
```

### Why split NIR/SWIR from MWIR/LWIR?

The two groups are governed by **different physics**, so they read different
buffers and use different math:

- **`VIS` / `NIR_ACTIVE` / `SWIR_ACTIVE` are reflective.** What the sensor sees is
  illumination (ambient, or an active illuminator co-located with the camera)
  **reflected** off surfaces. Brightness is governed by **reflectivity** (albedo,
  specular, roughness, `N·V`), and an active source obeys **inverse-square
  distance falloff** — a far surface returns less light. Colour dependence fades
  as wavelength grows (VIS is colour-aware, SWIR is nearly colour-blind).
- **`MWIR` / `LWIR` are emissive.** The surface is the source: it **emits** thermal
  radiation set by its **emissivity** and **temperature**. There is **no
  inverse-square illumination falloff** (you are not lighting the scene), geometry
  matters only weakly, and the signal is driven by heat (emissive materials,
  motion-derived heating, ambient temperature). MWIR sits on the steep part of the
  Planck curve so it favours **hot** targets (engines, exhausts); LWIR is where
  ambient-temperature objects peak, so the **whole scene** glows and emissivity
  contrast dominates.

Mixing these into one "thermal vs not" switch would force one model to fake the
other; separating them keeps each model physically honest.

### How the bands differ, visually and physically

| | What lights up | Driven by | Distance falloff |
|---|---|---|---|
| `VIS` | Bright, colourful, view-facing surfaces | albedo colour, `N·V` | none |
| `NIR_ACTIVE` | Near, smooth, specular surfaces (tight glints) | reflectivity, distance² | strong |
| `SWIR_ACTIVE` | Near, reflective **materials** (colour-agnostic) | material reflectance, distance² | strong |
| `MWIR` | **Hot spots** only — emissive/moving objects pop | temperature (`T⁴`) | none |
| `LWIR` | **Everything**, by emissivity; rough matte = bright | emissivity, ambient temp | none |

Practically: a smooth metal plate is a bright glint in `NIR_ACTIVE`/`SWIR_ACTIVE`
but **dark** in `LWIR` (low emissivity); a warm matte object is dim under active
NIR but **bright** in `LWIR`; and only a genuinely hot/emissive or fast-moving
object stands out in `MWIR`.

## IR synthesis

`synthesize_ir(mode, camera_pos=None, ambient_temp=293.0)` returns a **float32
`(H, W)` array in `[0, 1]`** built from per-buffer data. `mode` is a band name
(case-insensitive). Required buffers are auto-attached on first call. Background /
no-hit ("miss") rays are forced to `0`.

```python
ir = camera.synthesize_ir("LWIR")                                  # thermal
ir = camera.synthesize_ir("MWIR", ambient_temp=300.0)              # hot-biased thermal
ir = camera.synthesize_ir("NIR_ACTIVE", camera_pos=np.array([0, 0, 5]))   # active reflective
ir = camera.synthesize_ir("VIS")                                   # visible reflectance
```

- `camera_pos` is the world-space illuminator / viewpoint origin for the active
  reflective bands. Per-pixel world positions are reconstructed from the distance
  buffer + the camera prim's pose, and the view vector points from each surface
  toward `camera_pos` (falls back to a fixed `+Z` direction if `camera_pos` is
  omitted or the camera pose is unavailable).
- `ambient_temp` (Kelvin) sets the baseline temperature for the **emissive** bands
  (`MWIR`, `LWIR`); it is ignored by the reflective bands.

The output is already normalized to `[0, 1]` (divided by its own max) and has **no
noise** — display or save it directly. Keep the raw float map for training; only
colorize when you want to look at it.

> **Deprecated aliases.** The old mode names still work for backward compatibility
> but are deprecated and print a one-time notice:
> `mode="thermal"` → **`LWIR`**, `mode="active_nir"` → **`NIR_ACTIVE`**. Use the
> canonical band names in new code.

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

## Writing your own band model

Each band is a small, self-contained method on `SuperCamera` — `_synth_vis`,
`_synth_nir_active`, `_synth_swir_active`, `_synth_mwir`, `_synth_lwir` in
[`super_camera.py`](exts/super.camera/super/camera/super_camera.py). They all
receive the sanitized per-pixel buffer dict `bufs` (keyed by `BufferType.name`,
already NaN/Inf-scrubbed) plus the precomputed `dot_n_v` (clamped `N·V`), and
each returns a non-negative `(H, W)` intensity map. The shared tail of
`_compute_ir()` then masks miss rays and normalizes to `[0, 1]`, so you only worry
about the physics.

**Retune a band** — edit its `_synth_*` method, or the tuning constants near the
top of the file (`_NIR_SPECULAR_GAIN`, `_MWIR_EMISSIVE_GAIN`,
`_LWIR_GEOMETRY_WEIGHT`, …). Shared material helpers (`_luma`, `_gray`,
`_emissivity`, `_emissive_heat`, `_motion_heat`) are reusable building blocks.

**Add a band** — three steps:

1. Add a `SpectralBand` entry to `SPECTRAL_BANDS` in `buffers.py` (name, wavelength
   range, reflective/emissive class, description).
2. Register the buffers it needs in `_BAND_BUFFERS` in `super_camera.py`.
3. Add a `_synth_<band>` method and a branch in the `_compute_ir` dispatcher.

```python
def _synth_myband(self, bufs, dot_n_v):
    # produce any non-negative (H, W) array from the buffers you registered
    return np.mean(bufs["EMISSIVE"][:, :, :3], axis=2) * dot_n_v
```

The buffers available to a model are exactly the ones registered for that band in
`_BAND_BUFFERS`; `DISTANCE_TO_OBJECT` and `NORMALS` are always present. Guard
optional ones (e.g. `EMISSIVE`, which is unregistered in some builds) with
`if "EMISSIVE" in bufs`.

## Isaac Lab / training-pipeline integration

Isaac Lab already advances the renderer each step via `world.step(render=True)`.
Inside such a loop, **do not** let Super Camera call `rep.orchestrator.step()` —
use the no-step read path instead:

```python
# In the no-step read() path, attach the band's buffers up front so they are
# rendered before the first read (LWIR needs the emissive/material set).
camera = SuperCamera(
    prim_path="/World/SuperCamera",
    resolution=(640, 480),
    buffers=[
        BufferType.DISTANCE_TO_OBJECT, BufferType.NORMALS,
        BufferType.DIFFUSE_ALBEDO, BufferType.SPECULAR_ALBEDO, BufferType.ROUGHNESS,
        BufferType.EMISSIVE, BufferType.MOTION_VECTORS,
    ],
)
camera.aim(position=(3, 0, 2), target=(0, 0, 0.5))

for step in range(num_steps):
    sim.step(render=True)                            # Isaac Lab renders
    ir = camera.synthesize_ir_from_render("LWIR")    # reads the rendered frame, no extra step
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
| Spectral Band | Band dropdown (`VIS` / `NIR_ACTIVE` / `SWIR_ACTIVE` / `MWIR` / `LWIR`) with a live wavelength + reflective/emissive description; Ambient Temp enabled for emissive bands, Camera Position for active bands |
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
ir = camera.synthesize_ir("LWIR")
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
