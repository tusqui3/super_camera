# Super Camera — Claude Development Notes

## What this is

Isaac Sim extension + standalone Python library that wraps `omni.replicator.core` annotators into a single `SuperCamera` class. Avoids boilerplate render-product / annotator wiring. The goal is **simulated infrared imagery from per-buffer data** — see `synthesize_ir()` (thermal LWIR + active NIR). IR output is a float32 `(H,W)` map in `[0,1]`; `SuperCamera.colorize(ir, "ironbow")` maps it to a uint8 `(H,W,3)` **ironbow** (thermal-camera-look) RGB image. A jet-colormap depth image (`synthesize()`, driven by `DISTANCE_TO_OBJECT` → `distance_to_camera`) is also provided. The extension surfaces all of this through an auto-loaded GUI panel, and the library exposes a no-orchestrator-step `read()` / `synthesize_ir_from_render()` path for Isaac Lab integration.

## Project layout

```
super_camera/
├── CLAUDE.md
├── buffers.py                          ← root copy (kept identical to ext)
├── super_camera.py                     ← root copy (kept identical to ext)
├── exts/
│   └── super.camera/
│       ├── config/extension.toml
│       └── super/camera/
│           ├── __init__.py
│           ├── buffers.py              ← canonical source for BufferType / BufferData
│           ├── super_camera.py         ← SuperCamera class + _jet_colormap / _ironbow_colormap
│           └── extension.py           ← Omniverse lifecycle (on_startup / on_shutdown)
└── standalone/
    ├── example.py                      ← requires isaacsim installed
    ├── example_mock.py                 ← numpy only, no Isaac Sim needed
    └── isaaclab_integration.py         ← Isaac Lab sim-loop example (read() path)
```

**Root copies vs extension copies:** the root `buffers.py` / `super_camera.py` exist for IDE navigation convenience. Keep them byte-for-byte identical to the extension versions. Changes go to both.

## Running

Do NOT compile or run locally — the workstation with Isaac Sim is the only valid runtime. Use `standalone/example_mock.py` for logic-only iteration (no Omniverse import needed).

```bash
# On Isaac Sim workstation:
python standalone/example.py

# Anywhere (mock mode, numpy only):
python standalone/example_mock.py
```

## Synthetic image generation

`SuperCamera.synthesize(max_distance, colormap)` reads `DISTANCE_TO_OBJECT` (`distance_to_camera`) and returns a `uint8 (H,W,3)` numpy array.

- **`distance_to_camera`** — Euclidean distance in metres from camera origin to each surface point. Shape `(H,W)` float32. Different from `distance_to_image_plane` (orthogonal/z-buffer depth).
- **No-hit / background pixels read back as `inf`** in this build (not `0`). The valid mask is therefore `np.isfinite(depth) & (depth > 0)` — if you only test `depth > 0`, `inf` background slips through and poisons the auto-max (`max → inf`). Background stays black.
- If **every** pixel is `inf`, the camera is aimed at empty space — see `aim()` / camera-pose below. This is the #1 cause of a "uniform ambient + noise → static after AGC" IR frame.
- Normalization: `depth / scale`, clipped to `[0,1]`. `scale` defaults to the **maximum value of the valid float32 depth buffer** (auto-scaling per frame) when `max_distance=None`. Pass an explicit `max_distance` to use a fixed scale instead.
- Colormaps: `"jet"` (blue=near → red=far, default), `"ironbow"` (thermal-camera look), or `"grayscale"`. `synthesize()` delegates to the shared `SuperCamera.colorize(norm, colormap)` static method.
- To add more colormaps, add a branch to `colorize()` in `super_camera.py` (and a stops table like `_IRONBOW_STOPS` if it's a LUT-style palette).

## Colormaps & `colorize()`

`SuperCamera.colorize(image, colormap="ironbow")` is the single entry point that maps any `[0,1]` float `(H,W)` array to a uint8 `(H,W,3)` RGB image. Used by both `synthesize()` (depth) and the GUI (IR). Colormaps:

- `"ironbow"` (default) — FLIR-style thermal palette: black → deep purple → magenta → red → orange → yellow → white. Defined by the `_IRONBOW_STOPS` control-point table (position + RGB) and `_ironbow_colormap`, which `np.interp`s each channel across the stops. **Retune the look by editing the stops table**, not the math.
- `"jet"` — `_jet_colormap` (blue→red), used as the depth default.
- `"grayscale"` — stacked luminance.

A colormap is a **fixed LUT**, not AGC — it does not reintroduce the percentile-stretch static problem (see "Output, noise & display").

## IR camera modes

`SuperCamera.synthesize_ir(mode, camera_pos, ambient_temp)` returns a `float32 (H,W)` array in **`[0,1]`** built purely from per-buffer geometry data — **no temperature / radiometry / semantic physics, no noise**. The model is intentionally a thin, distance-driven scaffold: the base on which you layer your own heuristic, not a calibrated sensor model. The final output is normalized in `_compute_ir` by its own max (`out / out.max()`), so it is always directly displayable. Required buffers are auto-attached on first call.

**Miss rays:** background / no-hit pixels (where `distance_to_camera` is non-finite / `inf`) are forced to `0` in both modes — that's the "miss ray" handling. The `background` mask (`~np.isfinite(distance)`) is computed once in `_compute_ir` and applied to the final output.

`ambient_temp` is currently **unused** (kept only so the GUI call signature doesn't change). The GUI's "Thermal LWIR" label is now a misnomer — that mode is the distance scaffold below.

```python
ir = camera.synthesize_ir(mode="thermal")
ir = camera.synthesize_ir(mode="active_nir", camera_pos=np.array([0, 0, 5]))
```

### Mode: `thermal` — distance scaffold (edit me)

The minimal per-buffer model and the intended place to write your own heuristic. Required buffer: `DISTANCE_TO_OBJECT` (`NORMALS` is still attached for `active_nir` but unused here).

- `ir_intensity = distance` for hit pixels (miss → `0`); the shared tail then divides by `out.max()` → **far objects brighter**, near darker, normalized to `[0,1]`.
- The `else` branch in `_compute_ir` is one line — replace it with whatever function of the per-buffer data you want; the tail handles the miss-mask and `[0,1]` normalization.
- **Heuristic-authoring comment block:** the `else` branch carries a large inline comment that enumerates **every buffer accessible at that point** (pixel buffers in `bufs[...]`, the pre-computed `normals`/`view_vec`/`dot_n_v`/`background` arrays, and the structured buffers via `get_buffer()`), with shapes/dtypes and example heuristics. To use a buffer not currently attached in thermal mode, add it to `_THERMAL_BUFFERS` or call `add_buffer()`. Keep that comment in sync if the buffer set changes.

### Mode: `active_nir` (Active Near-Infrared)

A richer per-buffer model: coaxial active NIR illuminator (light source = camera), PBR reflectivity with Blinn-Phong specular and inverse-square distance falloff. Required buffers: `DISTANCE_TO_OBJECT`, `NORMALS`, `DIFFUSE_ALBEDO`, `SPECULAR_ALBEDO`, `ROUGHNESS`.

- Diffuse-to-grayscale (BT.601 luma) gives base NIR reflectivity.
- Specular highlight uses `shininess = 1 / roughness` with `H = V` (coaxial approximation).
- `ir_intensity = (diffuse_gray * NdotV + specular) / distance²`, distance clipped to `[0.1, 100]` metres.
- Background (miss) → `0`.

### Output, noise & display

- `synthesize_ir` output is already normalized to `[0,1]` (by `out.max()`) and **has no noise**. The GUI (`_on_capture`) clips it (`np.clip(ir, 0, 1)`) and runs it through `SuperCamera.colorize(disp, "ironbow")` → uint8 `(H,W,3)` thermal RGB, which it both previews and saves. The colormap is a **fixed LUT, not AGC** — it maps each `[0,1]` value to a fixed colour, so it does NOT reintroduce the percentile-stretch static. No `to_display` stretch in the capture path.
- **Why no `to_display` in the GUI:** the renderer's `distance_to_camera` buffer carries tiny per-pixel variation. `to_display`'s percentile stretch zooms into the local min↔max of whatever is in frame, so over a near-flat region it amplified that micro-variation to full-scale **TV static**. That percentile AGC was the noise source even after the explicit Gaussian noise was removed. Showing the already-normalized `[0,1]` output directly keeps a flat surface flat.
- `SuperCamera.to_display(ir, low_pct=2, high_pct=98)` (percentile stretch) still exists for anyone who *wants* AGC downstream, but it is **no longer used by the GUI**. Do not reintroduce it into the capture path — it is what produced the static.

`camera_pos` is only used when `POINTCLOUD` is available as an `(H, W, 3)` array. Otherwise a `+Z` fallback view vector is used (camera looks along world +Z).

## Capture pipeline & runtime rules

These apply to all capture/synthesis, not just IR. They encode hard-won fixes — don't regress them.

### Sync vs async stepping (Kit vs standalone)

`rep.orchestrator.step()` is synchronous and is **only legal in a standalone Python workflow**. Inside Kit (any GUI button callback runs on the Kit event loop) it raises *"synchronous call ... may not be made from within kit"*. So the capture/synthesis API comes in two flavors:

- Sync: `capture()`, `synthesize_ir()` — use from `standalone/example.py`.
- Async: `capture_async()`, `synthesize_ir_async()` — use from inside the extension. These `await rep.orchestrator.step_async()`.

Both share `_collect()` (assembles `BufferData` from annotators) and `_compute_ir()` (the IR math), so the two paths never diverge. The GUI `_on_capture` schedules `_capture_async()` via `asyncio.ensure_future`; never call the sync variants from `extension.py`. (`synthesize()` / `get_buffer()` are still sync-only — they're not used by the GUI.)

- No-step: `read()`, `synthesize_ir_from_render()` — for **externally-driven loops (Isaac Lab)**. They call `_collect()` / `_compute_ir()` directly and **never** touch `rep.orchestrator`, because the host app already renders each frame (`world.step(render=True)` / `sim.render()`). Calling the sync/async stepping variants there would double-step the app. Required buffers must already be attached (pass them at construction or via `add_buffer()`); `read()` has the same `_reattach_all()` recovery as `capture()` but does not step afterward.

### Isaac Lab integration

`standalone/isaaclab_integration.py` is the reference pattern: construct `SuperCamera` once with the buffers your heuristic needs, `aim()` it, then each step call `sim.step(render=True)` followed by `camera.synthesize_ir_from_render(mode)`. SuperCamera attaches its own standalone render product and never sets the active viewport camera, so it coexists with Isaac Lab's own sensors. Feed the float32 `(H,W)` IR map straight into observations/rewards; only `colorize()` it when you want to look at it.

### Viewport isolation & camera prim

The capture path must NOT disturb Isaac Sim's main viewport. Rules:

- `_initialize()` creates only a **plain USD camera prim** (`UsdGeom.Camera.Define`) + a standalone `rep.create.render_product`. It does **not** instantiate `omni.isaac.sensor.Camera` — that wrapper hooks the viewport/SDG pipeline and creates a redundant render product, which caused corrupted frames in the main viewport.
- `omni.isaac.sensor.Camera` is created **lazily** via `_ensure_isaac_camera()`, only when `set_pose()` / `get_intrinsics()` are called (standalone only — the GUI never calls them).
- **Aiming the camera:** `aim(position, target, up=(0,0,1))` sets the camera prim's transform **directly via USD** (`UsdGeom.Xformable` + a look-at `Gf.Matrix4d`) — it does **not** instantiate `omni.isaac.sensor.Camera`, so it is GUI-safe and does not touch the viewport. World is **Z-up** (default `up = +Z`); the camera looks down its local −Z. Without an `aim()` (or a manual transform in the stage), a freshly-created prim sits at the world origin looking down −Z and typically sees nothing → all-`inf` distance.
- The GUI only ever opens its **own** viewport window ("Super Camera Preview") in `_on_open_viewport`; it never sets the active camera on the main viewport.
- Caveat: `rep.orchestrator.step()` renders the whole app, so a brief frame change in the main viewport during capture is inherent to the global orchestrator and not yet eliminated.

### Render warm-up (cold AOV buffers)

A single `rep.orchestrator.step()` immediately after attaching an annotator reads the AOV **before the renderer has produced valid data** — the buffers come back uninitialized/garbage, which surfaces as a pure-noise IR image (both modes look like static, because the IR math is fed garbage normals/albedo). `_attach()` therefore sets `self._warmup_steps = _WARMUP_STEPS` (default 16); the next `capture()` / `capture_async()` runs that many extra steps via `_step_sync()` / `_step_async()` before the read, then resets the counter to 0. Warm-up only triggers on the first capture after a buffer is attached (or after `_reattach_all`), so steady-state video capture is still one step per frame. If IR still looks like static, raise `_WARMUP_STEPS`.

### Buffer sanitization

Whatever the warm-up state, `_compute_ir()` defends against stray uninitialized values: it captures the `background` mask from `~np.isfinite(DISTANCE_TO_OBJECT)` first, then runs every floating-point buffer through `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)` so any NaN/Inf in normals/albedo/roughness/distance becomes `0` before the IR math. Background (miss) pixels are still forced to `0` in the output via the mask. This means a partially-cold buffer degrades to black pixels instead of poisoning the whole frame with static.

### Annotator detachment recovery

External events (opening a viewport, stage edits) can reset the SyntheticData pipeline and silently detach annotators from standalone render products → *"annotator distance_to_camera is not attached to any render products"*. `capture()` / `capture_async()` catch a failed read, call `_reattach_all()` (rebuilds the render product + re-attaches every buffer in `self.buffers`), step once more, and retry. `self.buffers` is the source of truth for desired buffers; `_annotators` is the live set.

### Unregistered annotators

Not every annotator name exists in every Isaac Sim build. `emissive` in particular is **not registered** in some 4.x builds. `_attach()` catches `AnnotatorRegistry.get_annotator` failures, prints a `[super.camera] annotator '<name>' not registered, skipping` line, and continues — the buffer is simply omitted. Thermal mode tolerates a missing `EMISSIVE` (it only adds emissive-texture heat, and is guarded by `if "EMISSIVE" in bufs`), so synthesis still works without it. If your build exposes emissive under a different string, find it in the annotator list printed by the original error and update `BufferType.EMISSIVE`'s value in `buffers.py` (sync both copies).

## Buffer reference

All buffers are defined in `BufferType`. `ANNOTATOR_MAP` is auto-generated (`{bt: bt.value for bt in BufferType}`). To use a buffer, add it to the `buffers=` list at construction or call `camera.add_buffer(BufferType.X)`.

### Pixel buffers — `get_data()` returns `np.ndarray`

| BufferType | Annotator string | Shape | dtype | Notes |
|---|---|---|---|---|
| `DISTANCE_TO_OBJECT` | `distance_to_camera` | `(H,W)` | float32 | Euclidean dist to surface, metres |
| `DEPTH` | `distance_to_image_plane` | `(H,W)` | float32 | Orthogonal/z-buffer depth, metres |
| `NORMALS` | `normals` | `(H,W,4)` | float32 | World-space XYZ + unused W |
| `RGB` | `rgb` | `(H,W,4)` | uint8 | RGBA |
| `DIFFUSE_ALBEDO` | `diffuse_albedo` | `(H,W,4)` | float32 | PBR diffuse color |
| `SPECULAR_ALBEDO` | `specular_albedo` | `(H,W,4)` | float32 | PBR specular color |
| `ROUGHNESS` | `roughness` | `(H,W)` | float32 | PBR roughness scalar |
| `EMISSIVE` | `emissive` | `(H,W,4)` | float32 | Emissive color |
| `MOTION_VECTORS` | `motion_vectors` | `(H,W,4)` | float32 | 2-D screen-space optical flow |

### Structured buffers — `get_data()` returns `dict`

`BufferData.data` holds the `"data"` key from the dict; `BufferData.metadata` holds `"info"`. `BufferData.is_pixel_buffer` is `False` for these.

| BufferType | Annotator string | dict["data"] shape | Notes |
|---|---|---|---|
| `SEMANTIC` | `semantic_segmentation` | `(H,W)` uint32 | `metadata` has `idToLabels` |
| `INSTANCE` | `instance_segmentation` | `(H,W)` uint32 | hierarchical to lowest labeled prim |
| `INSTANCE_ID` | `instance_id_segmentation` | `(H,W)` uint32 | per-leaf prim |
| `OCCLUSION` | `occlusion` | structured | per-instance occlusion ratio |
| `BBOX_2D_TIGHT` | `bounding_box_2d_tight` | structured array | tight AABB, pixel coords |
| `BBOX_2D_LOOSE` | `bounding_box_2d_loose` | structured array | loose AABB, pixel coords |
| `BBOX_3D` | `bounding_box_3d` | structured array | 3-D cuboid + world pose |
| `CAMERA_PARAMS` | `camera_params` | — | full dict: projection, view transform, focal length, etc. |
| `POINTCLOUD` | `pointcloud` | `(N,3)` float32 | world-space XYZ |
| `SKELETON` | `skeleton_data` | — | character joint positions dict |

## Key design rules

- `ANNOTATOR_MAP` is derived automatically — adding a new `BufferType` member is enough to register the annotator name. No need to edit the map manually.
- Structured buffer types are listed in `STRUCTURED_BUFFERS` (frozenset). Any buffer in that set gets `{"data": ..., "info": ...}` unpacking applied automatically in `capture()` / `get_buffer()`.
- `_mock_data()` returns zero ndarrays for pixel buffers and `{"data": np.array([]), "info": {}}` for structured ones. Used when `mock=True` or Isaac Sim is not importable.
- `mock=True` is set automatically if `omni.replicator.core` import fails. Useful for CI and local development.

## Adding a new buffer

1. Add a member to `BufferType` with the exact Isaac Sim annotator string as its value.
2. If it returns a dict rather than an ndarray, add it to `STRUCTURED_BUFFERS`.
3. If it's a pixel buffer, add its channel count to `MOCK_SHAPES` (`()` for scalar/single-channel, `(4,)` for RGBA).
4. Sync root copies with extension copies.

## GUI panel (extension.py)

When the extension loads, a **"Super Camera"** window appears automatically in the Isaac Sim UI. No scripting needed.

| Section | Controls |
|---|---|
| Camera Setup | Prim Path (default `/World/SuperCamera`), Width × Height, **Create Camera**, **Open Viewport** |
| IR Mode | Mode dropdown (Thermal LWIR / Active NIR); relevant fields enabled, others greyed |
| Thermal LWIR | Ambient Temp (K) — background temperature, default 293 K |
| Active NIR | Camera Position XYZ — world-space illuminator origin |
| IR Preview | Live ironbow (thermal) thumbnail updated after each capture (320×180, RGBA ByteImageProvider) |
| Output | Save Path — where the RGB PNG (or PPM fallback) is written |
| Buttons | **Capture IR Frame** — runs one frame; **Reset Camera** — destroys camera object and closes viewport |

**Behaviour:**
- **Create Camera** — immediately creates the USD Camera prim at the given prim path and wires up the render product. The prim appears in the stage hierarchy and behaves as a standard Omniverse camera.
- **Open Viewport** — calls `omni.kit.viewport.utility.create_viewport_window()` to open a docked viewport window showing the live camera view. Sets the active camera via `viewport_api.camera_path` (falls back to `set_active_camera()` if needed). Creates the camera first if it doesn't exist yet.
- **Capture IR Frame** — synthesizes the selected IR mode, colorizes it with the ironbow palette (`SuperCamera.colorize`), updates the IR Preview thumbnail, and saves the file.
- If prim path or resolution changes between operations, the old camera is destroyed and a new one is created automatically.
- Output is an **RGB** (ironbow) PNG via Pillow; falls back to **PPM** (`P6`) if Pillow is unavailable in the Isaac Sim Python env. `_update_preview` / `_save_ir` both take a uint8 `(H,W,3)` RGB array.
- Status line updates after every action or error.
- `_on_mode_changed` is called immediately after UI build so the correct fields are greyed from the start.

## Extension discovery (critical)

Omniverse finds the extension by importing the module named in `extension.toml` (`[[python.module]] name = "super.camera"` → `super/camera/__init__.py`), then scanning that module's namespace for a subclass of `omni.ext.IExt`. The `SuperCameraExtension` class lives in `extension.py`, so `__init__.py` MUST re-export it:

```python
from .extension import SuperCameraExtension
```

Without this import the extension enables without error but `on_startup` never runs — no window, no menu, nothing. Do not remove this import.

## Isaac Sim setup

- Extension path: add `<repo>/exts` in **Window → Extensions → ⚙ → Add path**, then enable **Super Camera**.
- Dependency extensions: `omni.replicator.core`, `omni.isaac.sensor`, `omni.syntheticdata`, `omni.kit.viewport.utility`.
- Tested on Isaac Sim 4.x (Omniverse Kit 106.x).
- `distance_to_camera` and `distance_to_image_plane` return `(H,W)` float32, not `(H,W,1)` — the old code had this wrong; mock shapes now reflect reality.
- The old `albedo` / `"albedo"` annotator name was incorrect; Isaac Sim uses `diffuse_albedo`. Fixed in this version.
