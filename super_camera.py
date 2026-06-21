from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .buffers import (
    BufferType, BufferData, ANNOTATOR_MAP, ANNOTATOR_FALLBACKS, STRUCTURED_BUFFERS, MOCK_SHAPES,
    SPECTRAL_BANDS, DEPRECATED_BAND_ALIASES,
)

_WARMUP_STEPS = 16

# RTX anti-aliasing op. DLSS (3) is an upscaler: it renders the scene at a lower
# internal resolution and upscales the final colour, but the PBR material AOVs come
# back at that internal render resolution while distance/normals come back at the
# upscaled output resolution. That split mismatched AOV shapes and left the SDG
# OgnSdPostRenderVarToHost node with an invalid render-product binding, flooding the
# log every frame during capture. Forcing a non-upscaling op (TAA) for the duration
# of a capture keeps every AOV at the render product's native resolution. The key /
# value are build-specific — adjust here if the flood persists on a given build.
_AA_OP_SETTING = "/rtx/post/aa/op"
_AA_OP_NO_UPSCALE = 1

try:
    import omni.replicator.core as rep
    from omni.isaac.sensor import Camera
    _OMNIVERSE_AVAILABLE = True
except ImportError:
    _OMNIVERSE_AVAILABLE = False


def _jet_colormap(norm: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(norm * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(norm * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(norm * 4.0 - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


# "Ironbow" / "iron" palette — the classic look of FLIR-style thermal cameras:
# black → deep purple → magenta → red → orange → yellow → white as intensity rises.
# Columns are: stop position in [0,1], then R, G, B in [0,255].
_IRONBOW_STOPS = np.array([
    [0.00,   0,   0,   0],
    [0.20,  35,   0,  80],
    [0.40, 125,  20, 120],
    [0.55, 200,  40,  70],
    [0.70, 240, 110,  20],
    [0.85, 250, 200,  40],
    [1.00, 255, 255, 255],
], dtype=np.float32)


def _ironbow_colormap(norm: np.ndarray) -> np.ndarray:
    x = np.clip(norm, 0.0, 1.0)
    pos = _IRONBOW_STOPS[:, 0]
    r = np.interp(x, pos, _IRONBOW_STOPS[:, 1])
    g = np.interp(x, pos, _IRONBOW_STOPS[:, 2])
    b = np.interp(x, pos, _IRONBOW_STOPS[:, 3])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


# Buffers each spectral band needs attached before synthesis. Reflective bands
# (VIS / NIR_ACTIVE / SWIR_ACTIVE) need the PBR appearance set; emissive bands
# (MWIR / LWIR) additionally need EMISSIVE + MOTION_VECTORS for the heat proxies.
# DISTANCE_TO_OBJECT and NORMALS are required by every band (background mask and
# N·V). EMISSIVE may be unregistered in some builds — _attach() skips it and the
# emissive models guard with `if "EMISSIVE" in bufs`.
_REFLECTIVE_BUFFERS: List[BufferType] = [
    BufferType.DISTANCE_TO_OBJECT,
    BufferType.NORMALS,
    BufferType.DIFFUSE_ALBEDO,
    BufferType.SPECULAR_ALBEDO,
    BufferType.ROUGHNESS,
]

_EMISSIVE_BUFFERS: List[BufferType] = _REFLECTIVE_BUFFERS + [
    BufferType.EMISSIVE,
    BufferType.MOTION_VECTORS,
]

_BAND_BUFFERS: Dict[str, List[BufferType]] = {
    "VIS": _REFLECTIVE_BUFFERS,
    "NIR_ACTIVE": _REFLECTIVE_BUFFERS,
    "SWIR_ACTIVE": _REFLECTIVE_BUFFERS,
    "MWIR": _EMISSIVE_BUFFERS,
    "LWIR": _EMISSIVE_BUFFERS,
}

# ── IR band-model tuning constants ───────────────────────────────────────────
# These set the relative weighting between terms; absolute scale is irrelevant
# because _compute_ir normalizes the final frame by its own max.
_AMBIENT_REF_K = 293.0          # reference ambient temperature (K) → proxy 1.0
_NIR_SPECULAR_GAIN = 2.0        # active-NIR specular boost over passive VIS
_SWIR_SPECULAR_GAIN = 3.0       # active-SWIR specular boost (material emphasis)
_MOTION_SCALE = 8.0             # screen-space motion (px) at which heat saturates
_MWIR_EMISSIVE_GAIN = 4.0       # MWIR weight on emissive-material heat
_MWIR_MOTION_GAIN = 2.0         # MWIR weight on motion-derived heat
_LWIR_EMISSIVE_GAIN = 1.0       # LWIR weight on emissive-material heat
_LWIR_MOTION_GAIN = 0.5         # LWIR weight on motion-derived heat
_LWIR_GEOMETRY_WEIGHT = 0.15    # LWIR weak geometry (N·V) influence

_DEPRECATION_NOTIFIED: set = set()


def _resolve_band(mode: str) -> str:
    """Resolve a mode string to a canonical band name in SPECTRAL_BANDS.

    Accepts canonical names (case-insensitive) and the deprecated aliases
    `thermal` → LWIR and `active_nir` → NIR_ACTIVE (with a one-time notice).
    """
    if not isinstance(mode, str):
        raise ValueError(f"IR mode must be a band-name string, got {type(mode).__name__}")
    key = mode.strip()
    if key in SPECTRAL_BANDS:
        return key
    if key.upper() in SPECTRAL_BANDS:
        return key.upper()
    alias = DEPRECATED_BAND_ALIASES.get(key.lower())
    if alias is not None:
        if key.lower() not in _DEPRECATION_NOTIFIED:
            _DEPRECATION_NOTIFIED.add(key.lower())
            print(f"[super.camera] IR mode '{mode}' is deprecated; mapping to '{alias}'. "
                  f"Use the canonical band name instead.")
        return alias
    valid = ", ".join(SPECTRAL_BANDS)
    aliases = ", ".join(DEPRECATED_BAND_ALIASES)
    raise ValueError(f"Unknown IR band '{mode}'. Valid bands: {valid}. Deprecated aliases: {aliases}.")


class SuperCamera:

    def __init__(
        self,
        prim_path: str = "/World/SuperCamera",
        resolution: Tuple[int, int] = (1280, 720),
        buffers: Optional[List[BufferType]] = None,
        mock: bool = False,
    ):
        self.prim_path = prim_path
        self.resolution = resolution
        self.buffers = buffers or [BufferType.DISTANCE_TO_OBJECT]
        self._mock = mock or not _OMNIVERSE_AVAILABLE

        self._camera = None
        self._render_product = None
        self._annotators: Dict[BufferType, object] = {}
        self._warmup_steps = 0

        if not self._mock:
            self._initialize()

    def _initialize(self):
        self._ensure_camera_prim()
        self._render_product = rep.create.render_product(self.prim_path, self.resolution)
        for buffer_type in self.buffers:
            self._attach(buffer_type)

    def _ensure_camera_prim(self):
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(self.prim_path).IsValid():
            UsdGeom.Camera.Define(stage, self.prim_path)

    def _ensure_isaac_camera(self):
        if self._camera is None and not self._mock:
            self._camera = Camera(prim_path=self.prim_path, resolution=self.resolution)
            self._camera.initialize()
        return self._camera

    def _attach(self, buffer_type: BufferType) -> bool:
        primary = ANNOTATOR_MAP.get(buffer_type)
        if not primary:
            raise ValueError(f"Unsupported buffer type: {buffer_type}")
        candidates = [primary] + ANNOTATOR_FALLBACKS.get(buffer_type, [])
        annotator = None
        last_exc = None
        for name in candidates:
            try:
                annotator = rep.AnnotatorRegistry.get_annotator(name)
                break
            except Exception as exc:
                last_exc = exc
        if annotator is None:
            print(f"[super.camera] no registered annotator for {buffer_type.name} "
                  f"(tried {candidates}), skipping: {last_exc}")
            return False
        annotator.attach([self._render_product])
        self._annotators[buffer_type] = annotator
        self._warmup_steps = _WARMUP_STEPS
        return True

    def _detach(self, buffer_type: BufferType):
        annotator = self._annotators.pop(buffer_type, None)
        if annotator:
            annotator.detach([self._render_product])

    def _mock_data(self, buffer_type: BufferType) -> Any:
        if buffer_type in STRUCTURED_BUFFERS:
            return {"data": np.array([]), "info": {}}
        h, w = self.resolution[1], self.resolution[0]
        extra = MOCK_SHAPES.get(buffer_type, (4,))
        shape = (h, w) + extra if extra else (h, w)
        dtype = np.uint8 if buffer_type == BufferType.RGB else np.float32
        return np.zeros(shape, dtype=dtype)

    def _reattach_all(self):
        desired = list(self.buffers)
        for bt in list(self._annotators):
            try:
                self._annotators[bt].detach([self._render_product])
            except Exception:
                pass
        self._annotators.clear()
        try:
            if self._render_product is not None:
                self._render_product.destroy()
        except Exception:
            pass
        self._render_product = rep.create.render_product(self.prim_path, self.resolution)
        for bt in desired:
            self._attach(bt)

    def _wrap(self, bt: BufferType, raw: Any) -> BufferData:
        metadata = None
        if isinstance(raw, dict):
            metadata = raw.get("info")
            data = raw.get("data", raw)
        else:
            data = raw
        return BufferData(buffer_type=bt, data=data, metadata=metadata)

    def _collect(self) -> Dict[BufferType, BufferData]:
        if self._mock:
            return {bt: self._wrap(bt, self._mock_data(bt)) for bt in self.buffers}
        return {bt: self._wrap(bt, self._annotators[bt].get_data()) for bt in list(self._annotators)}

    def _consume_warmup_steps(self) -> int:
        n = self._warmup_steps
        self._warmup_steps = 0
        return n

    def _set_no_upscale(self) -> Any:
        # Switch the renderer to a non-upscaling AA op while the standalone render
        # product renders, returning the previous value so it can be restored. See
        # the _AA_OP_* constants for why upscaling (DLSS) breaks the AOV bindings.
        if self._mock:
            return None
        try:
            import carb
            settings = carb.settings.get_settings()
            prev = settings.get(_AA_OP_SETTING)
            if prev != _AA_OP_NO_UPSCALE:
                settings.set(_AA_OP_SETTING, _AA_OP_NO_UPSCALE)
            return prev
        except Exception:
            return None

    def _restore_aa(self, prev: Any):
        # Restore the AA op captured by _set_no_upscale so the main viewport's
        # appearance is left unchanged outside of a capture.
        if self._mock or prev is None:
            return
        try:
            import carb
            carb.settings.get_settings().set(_AA_OP_SETTING, prev)
        except Exception:
            pass

    def _step_sync(self):
        prev = self._set_no_upscale()
        try:
            for _ in range(self._consume_warmup_steps()):
                rep.orchestrator.step(rt_subframes=1)
            rep.orchestrator.step(rt_subframes=1)
        finally:
            self._restore_aa(prev)

    async def _step_async(self):
        prev = self._set_no_upscale()
        try:
            for _ in range(self._consume_warmup_steps()):
                await rep.orchestrator.step_async(rt_subframes=1)
            await rep.orchestrator.step_async(rt_subframes=1)
        finally:
            self._restore_aa(prev)

    def capture(self) -> Dict[BufferType, BufferData]:
        if self._mock:
            return self._collect()
        self._step_sync()
        try:
            return self._collect()
        except Exception as exc:
            print(f"[super.camera] read failed ({exc}); re-attaching annotators")
            self._reattach_all()
            self._step_sync()
            return self._collect()

    async def capture_async(self) -> Dict[BufferType, BufferData]:
        if self._mock:
            return self._collect()
        await self._step_async()
        try:
            return self._collect()
        except Exception as exc:
            print(f"[super.camera] read failed ({exc}); re-attaching annotators")
            self._reattach_all()
            await self._step_async()
            return self._collect()

    def read(self) -> Dict[BufferType, BufferData]:
        # Collect buffers from the frame the app has ALREADY rendered, without
        # calling rep.orchestrator.step(). Use this inside an externally-driven
        # simulation loop (e.g. Isaac Lab, where world.step(render=True) already
        # advances the renderer) so SuperCamera never double-steps the app.
        if self._mock:
            return self._collect()
        try:
            return self._collect()
        except Exception as exc:
            print(f"[super.camera] read failed ({exc}); re-attaching annotators")
            self._reattach_all()
            return self._collect()

    def synthesize_ir_from_render(
        self,
        mode: str = "LWIR",
        ambient_temp: float = 293.0,
    ) -> np.ndarray:
        # Like synthesize_ir(), but reads the already-rendered frame via read()
        # instead of stepping the orchestrator. Required buffers must already be
        # attached (pass them to buffers= at construction or call add_buffer()).
        band = _resolve_band(mode)
        for bt in _BAND_BUFFERS[band]:
            self.add_buffer(bt)
        captured = self.read()
        return self._compute_ir(captured, band, ambient_temp)

    def get_buffer(self, buffer_type: BufferType) -> BufferData:
        if not self._mock and buffer_type not in self._annotators:
            raise ValueError(f"Buffer '{buffer_type}' is not attached.")

        if not self._mock:
            rep.orchestrator.step(rt_subframes=1)

        raw = self._mock_data(buffer_type) if self._mock else self._annotators[buffer_type].get_data()
        metadata = None
        if isinstance(raw, dict):
            metadata = raw.get("info")
            data = raw.get("data", raw)
        else:
            data = raw
        return BufferData(buffer_type=buffer_type, data=data, metadata=metadata)

    def synthesize(
        self,
        max_distance: Optional[float] = None,
        colormap: str = "jet",
    ) -> np.ndarray:
        buf = self.get_buffer(BufferType.DISTANCE_TO_OBJECT)
        depth = buf.data.squeeze().astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        norm = np.zeros_like(depth)
        if valid.any():
            scale = max_distance if max_distance is not None else float(depth[valid].max())
            if scale > 0.0:
                norm[valid] = np.clip(depth[valid] / scale, 0.0, 1.0)
        return self.colorize(norm, colormap)

    @staticmethod
    def colorize(image: np.ndarray, colormap: str = "ironbow") -> np.ndarray:
        norm = np.clip(image.astype(np.float32), 0.0, 1.0)
        if colormap == "grayscale":
            gray = (norm * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)
        if colormap == "jet":
            return _jet_colormap(norm)
        return _ironbow_colormap(norm)

    def synthesize_ir(
        self,
        mode: str = "LWIR",
        ambient_temp: float = 293.0,
    ) -> np.ndarray:
        band = _resolve_band(mode)
        for bt in _BAND_BUFFERS[band]:
            self.add_buffer(bt)
        captured = self.capture()
        return self._compute_ir(captured, band, ambient_temp)

    async def synthesize_ir_async(
        self,
        mode: str = "LWIR",
        ambient_temp: float = 293.0,
    ) -> np.ndarray:
        band = _resolve_band(mode)
        for bt in _BAND_BUFFERS[band]:
            self.add_buffer(bt)
        captured = await self.capture_async()
        return self._compute_ir(captured, band, ambient_temp)

    def _compute_ir(
        self,
        captured: Dict[BufferType, BufferData],
        mode: str,
        ambient_temp: float,
    ) -> np.ndarray:
        # Resolve defensively so a direct _compute_ir() call (or an alias) still
        # selects the right band; the public callers already pass a canonical name.
        band = _resolve_band(mode)
        bufs = {bt.name: captured[bt].data for bt in captured}

        H, W = bufs["DISTANCE_TO_OBJECT"].shape
        background = ~np.isfinite(bufs["DISTANCE_TO_OBJECT"])

        # AOVs can come back at different resolutions (e.g. material AOVs at the
        # render resolution while distance/normals are at the upscaled output
        # resolution). Resample every pixel buffer to the distance buffer's (H, W)
        # so the per-pixel math broadcasts; distance defines the output size/mask.
        for _name, _arr in list(bufs.items()):
            bufs[_name] = self._resample_to(_arr, H, W)

        for _name, _arr in list(bufs.items()):
            if isinstance(_arr, np.ndarray) and np.issubdtype(_arr.dtype, np.floating):
                bufs[_name] = np.nan_to_num(_arr, nan=0.0, posinf=0.0, neginf=0.0)

        normals = bufs["NORMALS"][:, :, :3]
        norm_lens = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = np.divide(normals, norm_lens, out=np.zeros_like(normals), where=norm_lens > 0)

        # The illuminator is coaxial with the camera (the light sits at the camera
        # eye), so the fixed +Z view vector also serves as the illumination
        # direction for the N·V geometry term.
        view_vec = np.zeros((H, W, 3), dtype=np.float32)
        view_vec[:, :, 2] = 1.0

        dot_n_v = np.clip(np.sum(normals * view_vec, axis=2), 0.0, 1.0)

        # ── Per-band synthesis ───────────────────────────────────────────────
        # Each model below consumes `bufs` (raw per-pixel buffers, keyed by
        # BufferType.name and already NaN/Inf-sanitized) plus the derived
        # `dot_n_v` (clamped N·V) and returns a non-negative (H, W) intensity
        # map. The shared tail then forces miss/background pixels to 0 and
        # normalizes the frame to [0, 1]. See each _synth_* method for the
        # physical model and the buffers it reads.
        if band == "VIS":
            ir_intensity = self._synth_vis(bufs, dot_n_v)
        elif band == "NIR_ACTIVE":
            ir_intensity = self._synth_nir_active(bufs, dot_n_v)
        elif band == "SWIR_ACTIVE":
            ir_intensity = self._synth_swir_active(bufs, dot_n_v)
        elif band == "MWIR":
            ir_intensity = self._synth_mwir(bufs, ambient_temp)
        else:  # LWIR
            ir_intensity = self._synth_lwir(bufs, dot_n_v, ambient_temp)

        out = np.maximum(ir_intensity, 0.0).astype(np.float32)
        # Final guard: any inf/nan a synth model produced (e.g. an overflowed
        # specular multiply) becomes 0 here, so it can't poison out.max() (which
        # would collapse the whole-frame normalization) or leak into colorize().
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out[background] = 0.0
        omax = float(out.max())
        if omax > 0.0:
            out = out / omax

        return out

    # ── Shared appearance / material helpers ────────────────────────────────
    # `bufs` here is the sanitized per-pixel buffer dict from _compute_ir.
    #   bufs["DISTANCE_TO_OBJECT"] (H,W)   float32  Euclidean dist, metres
    #   bufs["NORMALS"]            (H,W,4) float32  world-space normal XYZ + pad
    #   bufs["DIFFUSE_ALBEDO"]     (H,W,4) float32  PBR diffuse/base colour RGBA
    #   bufs["SPECULAR_ALBEDO"]    (H,W,4) float32  PBR specular colour RGBA
    #   bufs["ROUGHNESS"]          (H,W)   float32  PBR roughness [0,1]
    #   bufs["EMISSIVE"]           (H,W,4) float32  emissive colour RGBA (optional)
    #   bufs["MOTION_VECTORS"]     (H,W,4) float32  screen-space optical flow (optional)
    # To add a band, append a model below, register its buffers in _BAND_BUFFERS,
    # and add the SpectralBand entry in buffers.py.

    @staticmethod
    def _luma(rgba: np.ndarray) -> np.ndarray:
        # BT.601 perceptual luma — colour-aware grayscale.
        return 0.299 * rgba[:, :, 0] + 0.587 * rgba[:, :, 1] + 0.114 * rgba[:, :, 2]

    @staticmethod
    def _gray(rgba: np.ndarray) -> np.ndarray:
        # Flat, colour-agnostic grayscale (channel mean).
        return np.mean(rgba[:, :, :3], axis=2)

    # The PBR material AOVs (diffuse_albedo / specular_albedo / roughness) are not
    # registered in every Isaac Sim build — _attach() skips any it can't get, so
    # they may be absent from `bufs`. These accessors fall back to neutral
    # constants so a band degrades to plain shading instead of raising KeyError.

    @staticmethod
    def _diffuse_gray(bufs: Dict[str, np.ndarray], color_aware: bool = True) -> Any:
        # Base reflectance from diffuse albedo; mid-gray 0.5 when unavailable.
        if "DIFFUSE_ALBEDO" in bufs:
            a = bufs["DIFFUSE_ALBEDO"]
            if a.ndim == 2:
                return a
            return SuperCamera._luma(a) if color_aware else SuperCamera._gray(a)
        return 0.5

    @staticmethod
    def _specular_gray(bufs: Dict[str, np.ndarray]) -> Any:
        # Specular reflectance; 0 (no highlights) when unavailable. Clamped to the
        # physical [0,1] range so a cold/garbage AOV (huge finite values that
        # nan_to_num leaves untouched) can't overflow the specular multiply.
        if "SPECULAR_ALBEDO" in bufs:
            a = bufs["SPECULAR_ALBEDO"]
            spec = a if a.ndim == 2 else SuperCamera._gray(a)
            return np.clip(spec, 0.0, 1.0)
        return 0.0

    @staticmethod
    def _roughness(bufs: Dict[str, np.ndarray], lo: float = 0.0) -> Any:
        # PBR roughness clamped to [lo, 1]; mid-roughness 0.5 when unavailable.
        # The Roughness AOV may come back single-channel (H,W) or packed (H,W,C).
        if "ROUGHNESS" in bufs:
            r = bufs["ROUGHNESS"]
            if r.ndim == 3:
                r = r[:, :, 0]
            return np.clip(r, lo, 1.0)
        return 0.5

    @staticmethod
    def _emissivity(bufs: Dict[str, np.ndarray]) -> np.ndarray:
        # Infer emissivity from PBR material: rough, diffuse (dielectric) surfaces
        # emit efficiently (ε → 1); smooth, specular (metallic) surfaces emit
        # poorly (ε → 0). Used by the emissive thermal bands.
        roughness = SuperCamera._roughness(bufs, lo=0.0)
        specular_gray = np.clip(SuperCamera._specular_gray(bufs), 0.0, 1.0)
        return np.clip(0.5 + 0.5 * roughness - 0.4 * specular_gray, 0.05, 1.0)

    @staticmethod
    def _emissive_heat(bufs: Dict[str, np.ndarray]) -> Any:
        # Self-lit / hot materials act as a heat source. 0 when EMISSIVE is
        # unattached or unregistered in this build. The EmissionAndForegroundMask
        # AOV packs emission RGB (+ a foreground mask) — take the colour channels.
        if "EMISSIVE" in bufs:
            e = bufs["EMISSIVE"]
            return e if e.ndim == 2 else SuperCamera._gray(e)
        return 0.0

    @staticmethod
    def _motion_heat(bufs: Dict[str, np.ndarray]) -> Any:
        # Screen-space motion magnitude as a friction/heating proxy, saturating
        # at _MOTION_SCALE px → [0, 1]. 0 when MOTION_VECTORS is unattached.
        if "MOTION_VECTORS" in bufs:
            mag = np.linalg.norm(bufs["MOTION_VECTORS"][:, :, :2], axis=2)
            return np.clip(mag / _MOTION_SCALE, 0.0, 1.0)
        return 0.0

    # ── Buffer geometry ─────────────────────────────────────────────────────

    @staticmethod
    def _resample_to(arr: Any, H: int, W: int) -> Any:
        # Nearest-neighbour resample of a pixel buffer to (H, W). Used so AOVs that
        # come back at a different resolution than the distance buffer still
        # broadcast in the per-pixel IR math. Non-array / sub-2-D inputs pass
        # through untouched.
        if not isinstance(arr, np.ndarray) or arr.ndim < 2:
            return arr
        h, w = arr.shape[:2]
        if h == H and w == W:
            return arr
        ys = (np.arange(H) * h // H).astype(np.intp)
        xs = (np.arange(W) * w // W).astype(np.intp)
        return arr[ys][:, xs]

    # ── Per-band synthesis models ───────────────────────────────────────────

    def _synth_vis(self, bufs: Dict[str, np.ndarray], dot_n_v: np.ndarray) -> np.ndarray:
        # VIS (400–700 nm, reflective). Passive ambient reflectance: colour-aware
        # diffuse Lambertian term + a specular sheen whose tightness grows as the
        # surface smooths (shininess = 1/roughness). No active illuminator, so no
        # inverse-square distance falloff.
        diffuse_gray = self._diffuse_gray(bufs, color_aware=True)
        specular_gray = self._specular_gray(bufs)
        roughness = self._roughness(bufs, lo=0.01)
        specular = specular_gray * (dot_n_v ** (1.0 / roughness))
        return diffuse_gray * dot_n_v + specular

    def _synth_nir_active(self, bufs: Dict[str, np.ndarray], dot_n_v: np.ndarray) -> np.ndarray:
        # NIR_ACTIVE (700–1000 nm, reflective). Coaxial active illuminator (light
        # source = camera, so H = V and N·H = N·V). Diffuse reflectance plus a
        # specular term boosted above VIS so smooth, low-roughness surfaces give
        # concentrated glints; inverse-square distance attenuation for the active
        # source.
        diffuse_gray = self._diffuse_gray(bufs, color_aware=True)
        specular_gray = self._specular_gray(bufs)
        roughness = self._roughness(bufs, lo=0.01)
        specular = _NIR_SPECULAR_GAIN * specular_gray * (dot_n_v ** (1.0 / roughness))
        reflection = diffuse_gray * dot_n_v + specular
        distance = np.clip(bufs["DISTANCE_TO_OBJECT"], 0.1, 100.0)
        return reflection / (distance ** 2)

    def _synth_swir_active(self, bufs: Dict[str, np.ndarray], dot_n_v: np.ndarray) -> np.ndarray:
        # SWIR_ACTIVE (1000–2500 nm, reflective). Like NIR_ACTIVE but largely
        # colour-blind: flat grayscale reflectance (channel mean, not luma) with
        # greater emphasis on material reflectance — a stronger specular term and
        # a diffuse base modulated by surface smoothness (1 - roughness). Active
        # source → inverse-square distance attenuation.
        reflect_gray = self._diffuse_gray(bufs, color_aware=False)
        specular_gray = self._specular_gray(bufs)
        roughness = self._roughness(bufs, lo=0.01)
        specular = _SWIR_SPECULAR_GAIN * specular_gray * (dot_n_v ** (1.0 / roughness))
        reflection = reflect_gray * dot_n_v * (1.0 - 0.5 * roughness) + specular
        distance = np.clip(bufs["DISTANCE_TO_OBJECT"], 0.1, 100.0)
        return reflection / (distance ** 2)

    def _synth_mwir(self, bufs: Dict[str, np.ndarray], ambient_temp: float) -> np.ndarray:
        # MWIR (3000–5000 nm, emissive). Thermal emission with a steep response.
        # Radiance ≈ emissivity · T_proxy^4 where the temperature proxy is the
        # ambient baseline plus strongly-weighted emissive-material and motion
        # heat. The T^4 power biases hard toward the hottest objects (engines,
        # exhausts) — cool ambient surfaces stay dim. Emission is ~isotropic, so
        # (unlike LWIR) there is no geometry term at all.
        emissivity = self._emissivity(bufs)
        ambient = max(ambient_temp, 1.0) / _AMBIENT_REF_K
        temp = ambient + _MWIR_EMISSIVE_GAIN * self._emissive_heat(bufs) \
            + _MWIR_MOTION_GAIN * self._motion_heat(bufs)
        return emissivity * (temp ** 4)

    def _synth_lwir(self, bufs: Dict[str, np.ndarray], dot_n_v: np.ndarray, ambient_temp: float) -> np.ndarray:
        # LWIR (8000–14000 nm, emissive). Ambient-temperature thermal emission:
        # every surface radiates at ~ambient, brightness set by emissivity, with
        # emissive-material and motion heat adding modest warmth. Near-linear in
        # temperature (so the whole scene is visible, not just hot spots),
        # geometry has only a weak influence, and there is NO inverse-square
        # falloff because this is emitted — not actively illuminated — radiation.
        emissivity = self._emissivity(bufs)
        ambient = max(ambient_temp, 1.0) / _AMBIENT_REF_K
        temp = ambient + _LWIR_EMISSIVE_GAIN * self._emissive_heat(bufs) \
            + _LWIR_MOTION_GAIN * self._motion_heat(bufs)
        geometry = (1.0 - _LWIR_GEOMETRY_WEIGHT) + _LWIR_GEOMETRY_WEIGHT * dot_n_v
        return emissivity * temp * geometry

    @staticmethod
    def to_display(ir: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
        ir = ir.astype(np.float32)
        lo = float(np.percentile(ir, low_pct))
        hi = float(np.percentile(ir, high_pct))
        if hi - lo < 1e-6:
            return np.zeros_like(ir)
        return np.clip((ir - lo) / (hi - lo), 0.0, 1.0)

    def add_buffer(self, buffer_type: BufferType):
        if buffer_type in self._annotators or (self._mock and buffer_type in self.buffers):
            return
        if not self._mock:
            if not self._attach(buffer_type):
                return
        self.buffers.append(buffer_type)

    def remove_buffer(self, buffer_type: BufferType):
        if not self._mock:
            self._detach(buffer_type)
        if buffer_type in self.buffers:
            self.buffers.remove(buffer_type)

    def set_pose(
        self,
        position: Tuple[float, float, float],
        target: Optional[Tuple[float, float, float]] = None,
    ):
        camera = self._ensure_isaac_camera()
        if camera:
            camera.set_world_pose(position=list(position))
            if target:
                camera.set_focus_distance(
                    float(np.linalg.norm(np.array(target) - np.array(position)))
                )

    def aim(
        self,
        position: Tuple[float, float, float],
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ):
        if self._mock:
            return
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        self._ensure_camera_prim()
        eye = np.asarray(position, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - eye
        fn = np.linalg.norm(forward)
        if fn == 0:
            return
        forward = forward / fn
        right = np.cross(forward, np.asarray(up, dtype=np.float64))
        rn = np.linalg.norm(right)
        if rn == 0:
            return
        right = right / rn
        true_up = np.cross(right, forward)
        m = Gf.Matrix4d(
            float(right[0]), float(right[1]), float(right[2]), 0.0,
            float(true_up[0]), float(true_up[1]), float(true_up[2]), 0.0,
            float(-forward[0]), float(-forward[1]), float(-forward[2]), 0.0,
            float(eye[0]), float(eye[1]), float(eye[2]), 1.0,
        )
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(self.prim_path))
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(m)

    def get_intrinsics(self) -> dict:
        camera = self._ensure_isaac_camera()
        if self._mock or not camera:
            w, h = self.resolution
            return {"focal_length": 24.0, "horizontal_aperture": 20.955, "resolution": (w, h)}
        return {
            "focal_length": camera.get_focal_length(),
            "horizontal_aperture": camera.get_horizontal_aperture(),
            "resolution": self.resolution,
        }

    def destroy(self):
        if not self._mock:
            for bt in list(self._annotators.keys()):
                self._detach(bt)
            if self._render_product:
                self._render_product.destroy()
                self._render_product = None
        self._camera = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()

    def __repr__(self) -> str:
        mode = "mock" if self._mock else "live"
        return f"SuperCamera(path={self.prim_path}, resolution={self.resolution}, buffers={[b.name for b in self.buffers]}, mode={mode})"
