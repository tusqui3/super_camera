# Super Camera — Claude Development Notes

## What this is

Isaac Sim extension + standalone Python library that wraps `omni.replicator.core` annotators into a single `SuperCamera` class. Avoids boilerplate render-product / annotator wiring. The goal is **simulated multispectral IR imagery from per-buffer data** — see `synthesize_ir(mode)`, which models **five spectral bands** identified by wavelength range: `VIS` (400–700 nm), `NIR_ACTIVE` (700–1000 nm), `SWIR_ACTIVE` (1000–2500 nm), `MWIR` (3000–5000 nm), `LWIR` (8000–14000 nm). The first three are **reflective** (illumination bouncing off surfaces); the last two are **emissive** (thermal radiation the surface emits). IR output is a float32 `(H,W)` map in `[0,1]`; `SuperCamera.colorize(ir, "ironbow")` maps it to a uint8 `(H,W,3)` **ironbow** (thermal-camera-look) RGB image. A jet-colormap depth image (`synthesize()`, driven by `DISTANCE_TO_OBJECT` → `distance_to_camera`) is also provided. The extension surfaces all of this through an auto-loaded GUI panel, and the library exposes a no-orchestrator-step `read()` / `synthesize_ir_from_render()` path for Isaac Lab integration.

The canonical band definitions live in `SPECTRAL_BANDS` (`buffers.py`) — a `SpectralBand` dataclass holding `name`, `wavelength_min_nm`, `wavelength_max_nm`, `reflective_vs_emissive`, `description` — and are the single source of truth for mode selection, the GUI dropdown, and the docs.

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
│           ├── buffers.py              ← canonical source for BufferType / BufferData / SpectralBand / SPECTRAL_BANDS
│           ├── super_camera.py         ← SuperCamera class + per-band _synth_* models + _jet/_ironbow colormaps
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

## IR spectral bands

`SuperCamera.synthesize_ir(mode, ambient_temp)` returns a `float32 (H,W)` array in **`[0,1]`** built purely from per-buffer render data — **no calibrated radiometry, no noise**. `mode` is a **spectral-band name** (case-insensitive); each band has its own physically-motivated `_synth_*` model. The final output is normalized in `_compute_ir` by its own max (`out / out.max()`), so it is always directly displayable. Required buffers are auto-attached on first call.

**Mode resolution & deprecated aliases:** `_resolve_band(mode)` maps the `mode` string to a canonical band in `SPECTRAL_BANDS`. It accepts canonical names (any case) plus two **deprecated** aliases kept for backward compatibility — `thermal` → **`LWIR`**, `active_nir` → **`NIR_ACTIVE`** (`DEPRECATED_BAND_ALIASES` in `buffers.py`) — printing a one-time `[super.camera]` deprecation notice. Public method defaults are now `mode="LWIR"`. Unknown names raise `ValueError`.

**Miss rays:** background / no-hit pixels (where `distance_to_camera` is non-finite / `inf`) are forced to `0` in every band — that's the "miss ray" handling. The `background` mask (`~np.isfinite(distance)`) is computed once in `_compute_ir` and applied to the final output.

**Buffers per band** (`_BAND_BUFFERS` in `super_camera.py`): reflective bands (`VIS`/`NIR_ACTIVE`/`SWIR_ACTIVE`) use `_REFLECTIVE_BUFFERS` = `DISTANCE_TO_OBJECT, NORMALS, DIFFUSE_ALBEDO, SPECULAR_ALBEDO, ROUGHNESS`. Emissive bands (`MWIR`/`LWIR`) use `_EMISSIVE_BUFFERS` = the reflective set **+** `EMISSIVE, MOTION_VECTORS`. `EMISSIVE` may be unregistered in some builds — `_attach()` skips it and the emissive models guard with `if "EMISSIVE" in bufs`.

**`ambient_temp`** (Kelvin) is now **used** by the emissive bands (`MWIR`/`LWIR`) as the baseline temperature proxy (`ambient_temp / _AMBIENT_REF_K`, ref 293 K). It is ignored by the reflective bands. The active reflective bands (`NIR_ACTIVE`/`SWIR_ACTIVE`) assume a **coaxial illuminator at the camera eye** (light shares the camera viewpoint) — there is no `camera_pos` / separate light-position input; the view direction doubles as the illumination direction (see "Coaxial illumination & view vector" below).

```python
ir = camera.synthesize_ir("LWIR")                     # thermal, ambient-dominated
ir = camera.synthesize_ir("MWIR", ambient_temp=300.0) # thermal, hot-biased
ir = camera.synthesize_ir("NIR_ACTIVE")               # active reflective
ir = camera.synthesize_ir("VIS")                      # passive visible reflectance
```

### Architecture: dispatch + per-band models

`_compute_ir` computes the shared per-pixel terms once — `background`, **per-buffer resampling to the distance resolution** (`_resample_to`, see below), NaN/Inf sanitization of every float buffer, the unit `normals`, the fixed coaxial `view_vec` (`+Z`), and `dot_n_v` (clamped `N·V`) — then **dispatches by band** to a `_synth_<band>(...)` method. Each model returns a non-negative `(H,W)` map; the shared tail forces `background` → 0 and normalizes to `[0,1]`. **Edit a band by editing its `_synth_*` method**; tune cross-band weighting via the `_*_GAIN` / `_*_WEIGHT` constants near the top of the file. Shared building blocks: `_luma` (BT.601), `_gray` (flat channel mean), `_emissivity`, `_emissive_heat`, `_motion_heat`. To add a band: add a `SpectralBand` to `SPECTRAL_BANDS`, register buffers in `_BAND_BUFFERS`, add a `_synth_*` method + a dispatcher branch.

### Reflective bands (illumination return)

- **`VIS` (400–700 nm):** passive ambient reflectance, colour-aware. `diffuse_gray = luma(DIFFUSE_ALBEDO)`, `specular = gray(SPECULAR_ALBEDO) · (N·V)^(1/roughness)`; `ir = diffuse_gray·N·V + specular`. **No** distance falloff (no active source).
- **`NIR_ACTIVE` (700–1000 nm):** coaxial active illuminator (`H = V`). Like VIS but specular is boosted by `_NIR_SPECULAR_GAIN` (smooth surfaces glint) and the whole reflection is divided by `distance²` (`DISTANCE_TO_OBJECT` clipped `[0.1,100]`).
- **`SWIR_ACTIVE` (1000–2500 nm):** colour-blind variant. Uses flat `gray(DIFFUSE_ALBEDO)` (not luma), diffuse base modulated by `(1 − 0.5·roughness)`, specular boosted by the larger `_SWIR_SPECULAR_GAIN` (greater material emphasis), inverse-square falloff.

### Emissive bands (thermal emission)

- **`MWIR` (3000–5000 nm):** `ir = ε · T⁴ · τ`. `ε = _emissivity()` (rough/diffuse → high, smooth/specular → low). `T = ambient + _MWIR_EMISSIVE_GAIN·emissive_heat + _MWIR_MOTION_GAIN·motion_heat`. The `T⁴` power **biases hard toward hot objects**; cool ambient stays dim. Emission is ~isotropic → **no geometry term** (`_synth_mwir` takes no `dot_n_v`).
- **`LWIR` (8000–14000 nm):** `ir = ε · T · geom · τ`. Near-**linear** in temperature so the whole scene glows; `T = ambient + _LWIR_EMISSIVE_GAIN·emissive_heat + _LWIR_MOTION_GAIN·motion_heat`; `geom = (1−_LWIR_GEOMETRY_WEIGHT) + _LWIR_GEOMETRY_WEIGHT·N·V` (weak geometry). **No** inverse-square falloff (emitted, not illuminated).

Both emissive bands multiply by an **atmospheric transmittance** `τ = _atmospheric_transmittance() = exp(−_ATMOSPHERIC_EXTINCTION · distance)` (Beer–Lambert; `DISTANCE_TO_OBJECT`, metres). `_ATMOSPHERIC_EXTINCTION` (default `0.02`/m) is **small on purpose** — it dims radiance only gradually with range so distant surfaces fade instead of all saturating to one colour, giving the background a depth gradient. This is the fix for "MWIR/LWIR too saturated, background not distinguishable": **raise** `_ATMOSPHERIC_EXTINCTION` for stronger near/far contrast, lower it toward `0` to disable. It applies **only to the emissive bands** — the reflective active bands already attenuate via inverse-square `1/d²`.

`_emissivity()` = `clip(0.5 + 0.5·roughness − 0.4·gray(specular), 0.05, 1.0)`. `_emissive_heat()` = `gray(EMISSIVE)` (0 if unattached). `_motion_heat()` = `clip(|motion_xy| / _MOTION_SCALE, 0, 1)` (0 if unattached).

### Why reflective (NIR/SWIR) and emissive (MWIR/LWIR) are separate

Different physics → different buffers and math. Reflective bands see **illumination reflected** off surfaces: brightness is governed by reflectivity (albedo/specular/roughness/`N·V`) and an active source obeys **inverse-square distance falloff**; colour dependence fades with wavelength (VIS colour-aware → SWIR nearly colour-blind). Emissive bands have the **surface as the source**: signal is set by **emissivity × temperature** with **no illumination falloff** and only weak geometry; MWIR sits on the steep part of the Planck curve (favours hot targets), LWIR is where ambient-temperature objects peak (whole scene glows, emissivity contrast dominates). Folding these into one switch would force one model to fake the other.

### Output, noise & display

- `synthesize_ir` output is already normalized to `[0,1]` (by `out.max()`) and **has no noise**. The GUI (`_on_capture`) clips it (`np.clip(ir, 0, 1)`) and runs it through `SuperCamera.colorize(disp, "ironbow")` → uint8 `(H,W,3)` thermal RGB, which it both previews and saves. The colormap is a **fixed LUT, not AGC** — it maps each `[0,1]` value to a fixed colour, so it does NOT reintroduce the percentile-stretch static. No `to_display` stretch in the capture path.
- **Why no `to_display` in the GUI:** the renderer's `distance_to_camera` buffer carries tiny per-pixel variation. `to_display`'s percentile stretch zooms into the local min↔max of whatever is in frame, so over a near-flat region it amplified that micro-variation to full-scale **TV static**. That percentile AGC was the noise source even after the explicit Gaussian noise was removed. Showing the already-normalized `[0,1]` output directly keeps a flat surface flat.
- `SuperCamera.to_display(ir, low_pct=2, high_pct=98)` (percentile stretch) still exists for anyone who *wants* AGC downstream, but it is **no longer used by the GUI**. Do not reintroduce it into the capture path — it is what produced the static.

### Coaxial illumination & view vector

The illuminator is assumed **coaxial with the camera** — the light source sits at the camera eye, so for the active reflective bands the view direction and the illumination direction are the same (`H = V`, hence `N·H = N·V`). `_compute_ir` therefore uses a **fixed `+Z` `view_vec`** for the `dot_n_v` (`N·V`) geometry term: no `camera_pos` argument, no per-pixel world-position reconstruction, no camera-pose / intrinsics reads. This is the single, simple model used by every band.

There used to be a `camera_pos` parameter that reconstructed per-pixel world positions (`_view_vectors` / `_reconstruct_world_pos`) so the illuminator could sit anywhere; that path was **removed**. It added complexity for a feature that was never needed (the light is always taken to be at the camera) and was a source of shape bugs. If you ever want an off-axis illuminator again, reintroduce a per-pixel view-vector field — but keep it shape-locked to the distance buffer's `(H,W)`.

### AOV resolution harmonization (`_resample_to`)

Different render AOVs can come back at **different resolutions** in the same frame — most commonly the PBR material AOVs (`DiffuseAlbedo`/`SpecularAlbedo`/`Roughness`/`EmissionAndForegroundMask`) at the renderer's internal **render resolution** while `distance_to_camera` / `normals` are at the **upscaled output resolution** (DLSS/TAA upscaling). Mixing them in the per-pixel math raised `operands could not be broadcast together with shapes (360,640) (720,1280)`. `_compute_ir` now resamples **every** pixel buffer to the distance buffer's `(H,W)` via `_resample_to` (nearest-neighbour, integer index map, no SciPy) before any synthesis. The distance buffer defines the output size and the `background` mask, so everything is harmonized to it; a buffer already at `(H,W)` passes through untouched, and non-array / sub-2-D values are left alone.

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

A single `rep.orchestrator.step()` immediately after attaching an annotator reads the AOV **before the renderer has produced valid data** — the buffers come back uninitialized/garbage, which surfaces as a pure-noise IR image (any band looks like static, because the model is fed garbage normals/albedo). `_attach()` therefore sets `self._warmup_steps = _WARMUP_STEPS` (default 16); the next `capture()` / `capture_async()` runs that many extra steps via `_step_sync()` / `_step_async()` before the read, then resets the counter to 0. Warm-up only triggers on the first capture after a buffer is attached (or after `_reattach_all`), so steady-state video capture is still one step per frame. If IR still looks like static, raise `_WARMUP_STEPS`. **Keep `_WARMUP_STEPS` modest** — every warm-up step renders the standalone render product, so an oversized count also multiplies any per-frame render-pipeline log spam (see below).

### Upscaler / AOV resolution (the `invalid renderProduct input` flood)

The RTX **upscaler (DLSS)** renders the scene at a lower internal resolution and upscales the final colour. Under it, the PBR material AOVs (`DiffuseAlbedo` / `SpecularAlbedo` / `Roughness` / `EmissionAndForegroundMask`) come back at the **internal render resolution** while `distance_to_camera` / `normals` come back at the **upscaled output resolution**. That split caused two symptoms: (1) the `_compute_ir` shape-broadcast error fixed by `_resample_to`, and (2) a flood of `[omni.syntheticdata.plugin] OgnSdPostRenderVarToHost : invalid renderProduct input` — the SDG device-to-host node treating the mis-bound material render vars as an invalid render product, logged **once per attached annotator per rendered frame** during the capture's orchestrator stepping (so the warm-up steps multiply it). `_step_sync()` / `_step_async()` therefore call `_set_no_upscale()` before stepping and `_restore_aa()` in a `finally`: they force a **non-upscaling AA op** (`_AA_OP_SETTING` = `/rtx/post/aa/op` → `_AA_OP_NO_UPSCALE` = 1, TAA) for the duration of the render, then restore the previous value so the main viewport's appearance is left unchanged outside a capture. Both helpers are mock-safe and fully guarded (a wrong/unknown setting key degrades to a no-op). The key/value are **build-specific** — if the flood persists, adjust the `_AA_OP_*` constants (e.g. off `0`, or DLAA/RTXAA `4`, which also renders at native resolution). `_resample_to` stays as a safety net for any residual mismatch.

### Buffer sanitization

Whatever the warm-up state, `_compute_ir()` defends against stray uninitialized values: it captures the `background` mask from `~np.isfinite(DISTANCE_TO_OBJECT)` first, then runs every floating-point buffer through `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)` so any NaN/Inf in normals/albedo/roughness/distance becomes `0` before the IR math. Background (miss) pixels are still forced to `0` in the output via the mask. This means a partially-cold buffer degrades to black pixels instead of poisoning the whole frame with static.

### Annotator detachment recovery

External events (opening a viewport, stage edits) can reset the SyntheticData pipeline and silently detach annotators from standalone render products → *"annotator distance_to_camera is not attached to any render products"*. `capture()` / `capture_async()` catch a failed read, call `_reattach_all()` (rebuilds the render product + re-attaches every buffer in `self.buffers`), step once more, and retry. `self.buffers` is the source of truth for desired buffers; `_annotators` is the live set.

### Unregistered annotators

Not every annotator name exists in every Isaac Sim build. The PBR material AOVs are the main offenders: on **Kit 107.x** they are `DiffuseAlbedo` / `SpecularAlbedo` / `Roughness` / `EmissionAndForegroundMask` (CamelCase), but older builds used `diffuse_albedo` / … (lowercase). `_attach()` walks **primary → `ANNOTATOR_FALLBACKS`** (`buffers.py`) and uses the first the `AnnotatorRegistry` knows; if none resolve it prints `[super.camera] no registered annotator for <NAME> (tried [...]), skipping` and omits the buffer — it never reaches `bufs`. To discover the right string for a new build, read the `Available annotators: [...]` list in that skip message and add it to `ANNOTATOR_FALLBACKS` (or make it the `BufferType` value).

**All material buffers are accessed through guarded, shape-tolerant helpers** so a missing or oddly-shaped AOV degrades gracefully instead of raising `KeyError`/broadcast errors during synthesis: `_diffuse_gray()` → mid-gray `0.5`, `_specular_gray()` → `0.0` (no highlights), `_roughness()` → `0.5` (and reduces `(H,W,C)` → `(H,W)`), `_emissive_heat()` / `_motion_heat()` → `0.0`. A band whose material AOV is unavailable falls back to plain `N·V` shading (reflective) or uniform emissivity (emissive) rather than crashing. (Symptom this fixed: `Capture failed: 'ROUGHNESS'` — a `KeyError` from a synth model reading `bufs["ROUGHNESS"]` when the `roughness` annotator name was wrong for the build.)

## Buffer reference

All buffers are defined in `BufferType`. `ANNOTATOR_MAP` is auto-generated (`{bt: bt.value for bt in BufferType}`). To use a buffer, add it to the `buffers=` list at construction or call `camera.add_buffer(BufferType.X)`.

### Pixel buffers — `get_data()` returns `np.ndarray`

| BufferType | Annotator string | Shape | dtype | Notes |
|---|---|---|---|---|
| `DISTANCE_TO_OBJECT` | `distance_to_camera` | `(H,W)` | float32 | Euclidean dist to surface, metres |
| `DEPTH` | `distance_to_image_plane` | `(H,W)` | float32 | Orthogonal/z-buffer depth, metres |
| `NORMALS` | `normals` | `(H,W,4)` | float32 | World-space XYZ + unused W |
| `RGB` | `rgb` | `(H,W,4)` | uint8 | RGBA |
| `DIFFUSE_ALBEDO` | `DiffuseAlbedo` (RTX AOV) | `(H,W,4)` | float | PBR diffuse color |
| `SPECULAR_ALBEDO` | `SpecularAlbedo` (RTX AOV) | `(H,W,4)` | float | PBR specular color |
| `ROUGHNESS` | `Roughness` (RTX AOV) | `(H,W[,C])` | float | PBR roughness |
| `EMISSIVE` | `EmissionAndForegroundMask` (RTX AOV) | `(H,W,4)` | float | Emission color + foreground mask |
| `MOTION_VECTORS` | `motion_vectors` | `(H,W,4)` | float32 | 2-D screen-space optical flow |

The four PBR material AOVs use **CamelCase** annotator names on Kit 107.x; lowercase names (`diffuse_albedo`, …) from older builds are kept as `ANNOTATOR_FALLBACKS` (`buffers.py`) and `_attach()` tries primary → fallbacks. `Roughness` can come back single- or multi-channel — `_roughness()` reduces `(H,W,C)` to `(H,W)`. All four are read through guarded accessors, so an unavailable AOV degrades gracefully (see "Unregistered annotators").

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

## Adding a new spectral band

1. Add a `SpectralBand` entry to `SPECTRAL_BANDS` in `buffers.py` (`name`, `wavelength_min_nm`, `wavelength_max_nm`, `reflective_vs_emissive` = `REFLECTIVE`/`EMISSIVE`, `description`). This is the single source of truth — the GUI dropdown, `_resolve_band`, and the docs all read it.
2. Register the buffers the band needs in `_BAND_BUFFERS` in `super_camera.py` (reuse `_REFLECTIVE_BUFFERS` / `_EMISSIVE_BUFFERS` or define a new list). `DISTANCE_TO_OBJECT` + `NORMALS` are always required.
3. Add a `_synth_<band>(self, bufs, dot_n_v[, ambient_temp])` method returning a non-negative `(H,W)` map, and a branch in the `_compute_ir` dispatcher. Reuse the shared helpers (`_luma`, `_gray`, `_emissivity`, `_emissive_heat`, `_motion_heat`).
4. If the GUI should treat it as an active-illuminator band (shows the coaxial camera-eye light note), add its name to `_ACTIVE_BANDS` in `extension.py`. Emissive bands automatically get the ambient-temp field via `reflective_vs_emissive`.
5. Sync root copies with extension copies.

## GUI panel (extension.py)

When the extension loads, a **"Super Camera"** window appears automatically in the Isaac Sim UI. No scripting needed.

| Section | Controls |
|---|---|
| Camera Setup | Prim Path (default `/World/SuperCamera`), Width × Height, **Create Camera**, **Open Viewport** |
| Spectral Band | Band dropdown (`VIS` / `NIR_ACTIVE` / `SWIR_ACTIVE` / `MWIR` / `LWIR`, built from `SPECTRAL_BANDS`, default `LWIR`) + a live `_band_desc` label showing the wavelength range and reflective/emissive description; relevant fields enabled, others greyed |
| Ambient Temp (K) | Baseline temperature for the **emissive** bands (`MWIR`/`LWIR`), default 293 K — greyed for reflective bands |
| Coaxial-light note | A read-only label shown for the **active** bands (`NIR_ACTIVE`/`SWIR_ACTIVE`) stating the light source is located at the camera eye — greyed otherwise. No XYZ input (the old Camera Position fields were removed) |
| IR Preview | Live ironbow (thermal) thumbnail updated after each capture (320×180, RGBA ByteImageProvider) |
| Output | Save Path — where the RGB PNG (or PPM fallback) is written |
| Buttons | **Capture IR Frame** — runs one frame; **Reset Camera** — destroys camera object and closes viewport |

**Behaviour:**
- **Create Camera** — immediately creates the USD Camera prim at the given prim path and wires up the render product. The prim appears in the stage hierarchy and behaves as a standard Omniverse camera.
- **Open Viewport** — calls `omni.kit.viewport.utility.create_viewport_window()` to open a docked viewport window showing the live camera view. Sets the active camera via `viewport_api.camera_path` (falls back to `set_active_camera()` if needed). Creates the camera first if it doesn't exist yet.
- **Capture IR Frame** — synthesizes the selected band (`band = _BAND_NAMES[self._mode_idx]`), colorizes it with the ironbow palette (`SuperCamera.colorize`), updates the IR Preview thumbnail, and saves the file. `ambient_temp` is always passed (the model ignores it for reflective bands); there is no `camera_pos` (active bands use the coaxial camera-eye illuminator).
- If prim path or resolution changes between operations, the old camera is destroyed and a new one is created automatically.
- Output is an **RGB** (ironbow) PNG via Pillow; falls back to **PPM** (`P6`) if Pillow is unavailable in the Isaac Sim Python env. `_update_preview` / `_save_ir` both take a uint8 `(H,W,3)` RGB array.
- Status line updates after every action or error.
- `_on_mode_changed` is called immediately after UI build so the correct fields are greyed (and `_band_desc` populated) from the start. It enables Ambient Temp for emissive bands (`reflective_vs_emissive == EMISSIVE`) and the coaxial-light note for `_ACTIVE_BANDS`.

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
- PBR material AOV annotator names are **build-specific**: Kit 107.x uses CamelCase (`DiffuseAlbedo`, `SpecularAlbedo`, `Roughness`, `EmissionAndForegroundMask`); older builds used lowercase (`diffuse_albedo`, …), kept as `ANNOTATOR_FALLBACKS`. The legacy `albedo` name was wrong on every build.
