from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .buffers import (
    BufferType, BufferData, ANNOTATOR_MAP, STRUCTURED_BUFFERS, MOCK_SHAPES,
)

_WARMUP_STEPS = 16

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


_NIR_BUFFERS: List[BufferType] = [
    BufferType.DISTANCE_TO_OBJECT,
    BufferType.NORMALS,
    BufferType.DIFFUSE_ALBEDO,
    BufferType.SPECULAR_ALBEDO,
    BufferType.ROUGHNESS,
]

_THERMAL_BUFFERS: List[BufferType] = [
    BufferType.DISTANCE_TO_OBJECT,
    BufferType.NORMALS,
]


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
        annotator_name = ANNOTATOR_MAP.get(buffer_type)
        if not annotator_name:
            raise ValueError(f"Unsupported buffer type: {buffer_type}")
        try:
            annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
        except Exception as exc:
            print(f"[super.camera] annotator '{annotator_name}' not registered, skipping {buffer_type.name}: {exc}")
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

    def _step_sync(self):
        for _ in range(self._consume_warmup_steps()):
            rep.orchestrator.step(rt_subframes=1)
        rep.orchestrator.step(rt_subframes=1)

    async def _step_async(self):
        for _ in range(self._consume_warmup_steps()):
            await rep.orchestrator.step_async(rt_subframes=1)
        await rep.orchestrator.step_async(rt_subframes=1)

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
        mode: str = "thermal",
        camera_pos: Optional[np.ndarray] = None,
        ambient_temp: float = 293.0,
    ) -> np.ndarray:
        # Like synthesize_ir(), but reads the already-rendered frame via read()
        # instead of stepping the orchestrator. Required buffers must already be
        # attached (pass them to buffers= at construction or call add_buffer()).
        required = _NIR_BUFFERS if mode == "active_nir" else _THERMAL_BUFFERS
        for bt in required:
            self.add_buffer(bt)
        captured = self.read()
        return self._compute_ir(captured, mode, camera_pos, ambient_temp)

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
        mode: str = "thermal",
        camera_pos: Optional[np.ndarray] = None,
        ambient_temp: float = 293.0,
    ) -> np.ndarray:
        required = _NIR_BUFFERS if mode == "active_nir" else _THERMAL_BUFFERS
        for bt in required:
            self.add_buffer(bt)
        captured = self.capture()
        return self._compute_ir(captured, mode, camera_pos, ambient_temp)

    async def synthesize_ir_async(
        self,
        mode: str = "thermal",
        camera_pos: Optional[np.ndarray] = None,
        ambient_temp: float = 293.0,
    ) -> np.ndarray:
        required = _NIR_BUFFERS if mode == "active_nir" else _THERMAL_BUFFERS
        for bt in required:
            self.add_buffer(bt)
        captured = await self.capture_async()
        return self._compute_ir(captured, mode, camera_pos, ambient_temp)

    def _compute_ir(
        self,
        captured: Dict[BufferType, BufferData],
        mode: str,
        camera_pos: Optional[np.ndarray],
        ambient_temp: float,
    ) -> np.ndarray:
        bufs = {bt.name: captured[bt].data for bt in captured}

        H, W = bufs["DISTANCE_TO_OBJECT"].shape
        background = ~np.isfinite(bufs["DISTANCE_TO_OBJECT"])

        for _name, _arr in list(bufs.items()):
            if isinstance(_arr, np.ndarray) and np.issubdtype(_arr.dtype, np.floating):
                bufs[_name] = np.nan_to_num(_arr, nan=0.0, posinf=0.0, neginf=0.0)

        normals = bufs["NORMALS"][:, :, :3]
        norm_lens = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = np.divide(normals, norm_lens, out=np.zeros_like(normals), where=norm_lens > 0)

        if camera_pos is not None and "POINTCLOUD" in bufs and bufs["POINTCLOUD"].ndim == 3:
            view_vec = camera_pos - bufs["POINTCLOUD"][:, :, :3]
        else:
            view_vec = np.zeros((H, W, 3), dtype=np.float32)
            view_vec[:, :, 2] = 1.0

        view_lens = np.linalg.norm(view_vec, axis=2, keepdims=True)
        view_vec = np.divide(view_vec, view_lens, out=np.zeros_like(view_vec), where=view_lens > 0)

        dot_n_v = np.clip(np.sum(normals * view_vec, axis=2), 0.0, 1.0)

        if mode == "active_nir":
            diffuse = bufs["DIFFUSE_ALBEDO"]
            diffuse_gray = 0.299 * diffuse[:, :, 0] + 0.587 * diffuse[:, :, 1] + 0.114 * diffuse[:, :, 2]
            specular_gray = np.mean(bufs["SPECULAR_ALBEDO"][:, :, :3], axis=2)
            roughness = np.clip(bufs["ROUGHNESS"], 0.01, 1.0)
            specular_component = specular_gray * (dot_n_v ** (1.0 / roughness))
            reflection = diffuse_gray * dot_n_v + specular_component
            distance = np.clip(bufs["DISTANCE_TO_OBJECT"], 0.1, 100.0)
            ir_intensity = reflection / (distance ** 2)
        else:
            # ════════════════════════════════════════════════════════════════
            #  THERMAL MODE — WRITE YOUR OWN PER-BUFFER HEURISTIC HERE
            # ════════════════════════════════════════════════════════════════
            #
            #  `bufs` is a dict keyed by BufferType.name (a string) holding the
            #  raw per-pixel render data for this frame. Build whatever function
            #  of these arrays you like and assign the result to `ir_intensity`
            #  (a float (H, W) array). The shared tail below then:
            #    • forces miss/background pixels to 0 via `background`, and
            #    • normalizes the whole frame to [0, 1] by dividing by its max.
            #  So you only need to produce a non-negative (H, W) intensity map;
            #  scaling and miss-masking are handled for you.
            #
            #  ── WHICH BUFFERS ARE PRESENT ───────────────────────────────────
            #  Only ATTACHED buffers appear in `bufs`. Thermal mode attaches the
            #  set in _THERMAL_BUFFERS (DISTANCE_TO_OBJECT, NORMALS). To use any
            #  other buffer below, either add it to _THERMAL_BUFFERS at the top
            #  of this file, or call camera.add_buffer(BufferType.X) before
            #  synthesizing. Guard optional ones with e.g. `if "EMISSIVE" in bufs`.
            #  Every floating-point buffer here has already been run through
            #  np.nan_to_num (NaN/Inf → 0), so it is safe to do math on.
            #
            #  ── PIXEL BUFFERS (np.ndarray) — directly usable in the math ─────
            #   bufs["DISTANCE_TO_OBJECT"]  (H,W)   float32  Euclidean dist to surface, metres (miss = inf → see `background`)
            #   bufs["DEPTH"]               (H,W)   float32  Orthogonal/z-buffer depth to image plane, metres
            #   bufs["NORMALS"]             (H,W,4) float32  World-space surface normal XYZ (+ unused W); `normals` above is the unit XYZ
            #   bufs["RGB"]                 (H,W,4) uint8    Rendered colour RGBA
            #   bufs["DIFFUSE_ALBEDO"]      (H,W,4) float32  PBR diffuse/base colour RGBA
            #   bufs["SPECULAR_ALBEDO"]     (H,W,4) float32  PBR specular colour RGBA
            #   bufs["ROUGHNESS"]           (H,W)   float32  PBR roughness scalar [0,1]
            #   bufs["EMISSIVE"]            (H,W,4) float32  Emissive colour RGBA (self-lit materials — natural heat proxy)
            #   bufs["MOTION_VECTORS"]      (H,W,4) float32  Screen-space optical flow (friction/motion-heat proxy)
            #
            #  ── DERIVED VALUES ALREADY COMPUTED ABOVE (reuse freely) ─────────
            #   normals    (H,W,3) float32  unit world-space normal
            #   view_vec   (H,W,3) float32  unit camera→surface view direction
            #   dot_n_v    (H,W)   float32  clamped N·V (1 = facing camera, 0 = grazing)
            #   background (H,W)   bool     True where the ray missed all geometry
            #
            #  ── STRUCTURED BUFFERS (dicts / structured arrays, NOT pixel maps) ─
            #  These are NOT in `bufs` as plain images; access them with
            #  self.get_buffer(BufferType.X).data / .metadata if you attach them:
            #   SEMANTIC       (H,W) uint32  per-pixel class id  (metadata: idToLabels)
            #   INSTANCE       (H,W) uint32  hierarchical instance id
            #   INSTANCE_ID    (H,W) uint32  per-leaf-prim instance id
            #   OCCLUSION              per-instance occlusion ratio
            #   BBOX_2D_TIGHT          tight 2-D boxes (pixel coords)
            #   BBOX_2D_LOOSE          loose 2-D boxes (pixel coords)
            #   BBOX_3D                3-D oriented boxes + world pose
            #   CAMERA_PARAMS          intrinsics/extrinsics dict
            #   POINTCLOUD     (N,3) float32  world-space hit points
            #   SKELETON               character joint positions
            #
            #  Example heuristics you might write instead of the line below:
            #   • emissive heat:   ir_intensity = np.mean(bufs["EMISSIVE"][:,:,:3], axis=2)
            #   • class-based:     ir_intensity = np.where(sem == HUMAN_ID, 1.0, 0.2)
            #   • view-warmed:     ir_intensity = dot_n_v / np.clip(bufs["DISTANCE_TO_OBJECT"], 0.1, None)
            #
            #  CURRENT (default) BEHAVIOUR — distance scaffold: far = bright.
            #  Replace the next line with your heuristic; keep the result (H,W) ≥ 0.
            ir_intensity = np.where(background, 0.0, bufs["DISTANCE_TO_OBJECT"])

        out = np.maximum(ir_intensity, 0.0).astype(np.float32)
        out[background] = 0.0
        omax = float(out.max())
        if omax > 0.0:
            out = out / omax

        return out

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
