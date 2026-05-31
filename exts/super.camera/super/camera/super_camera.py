from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from .buffers import BufferType, BufferData, ANNOTATOR_MAP

try:
    import omni.replicator.core as rep
    from omni.isaac.sensor import Camera
    _OMNIVERSE_AVAILABLE = True
except ImportError:
    _OMNIVERSE_AVAILABLE = False


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
        self.buffers = buffers or [BufferType.RGB, BufferType.DEPTH]
        self._mock = mock or not _OMNIVERSE_AVAILABLE

        self._camera = None
        self._render_product = None
        self._annotators: Dict[BufferType, object] = {}

        if not self._mock:
            self._initialize()

    def _initialize(self):
        self._camera = Camera(prim_path=self.prim_path, resolution=self.resolution)
        self._camera.initialize()
        self._render_product = rep.create.render_product(self.prim_path, self.resolution)
        for buffer_type in self.buffers:
            self._attach(buffer_type)

    def _attach(self, buffer_type: BufferType):
        annotator_name = ANNOTATOR_MAP.get(buffer_type)
        if not annotator_name:
            raise ValueError(f"Unsupported buffer type: {buffer_type}")
        annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
        annotator.attach([self._render_product])
        self._annotators[buffer_type] = annotator

    def _detach(self, buffer_type: BufferType):
        annotator = self._annotators.pop(buffer_type, None)
        if annotator:
            annotator.detach([self._render_product])

    def _mock_data(self, buffer_type: BufferType) -> np.ndarray:
        h, w = self.resolution[1], self.resolution[0]
        shapes = {
            BufferType.RGB: (h, w, 4),
            BufferType.DEPTH: (h, w, 1),
            BufferType.NORMALS: (h, w, 4),
            BufferType.SEMANTIC: (h, w, 4),
            BufferType.INSTANCE: (h, w, 4),
            BufferType.MOTION_VECTORS: (h, w, 4),
            BufferType.OCCLUSION: (h, w, 1),
            BufferType.ALBEDO: (h, w, 4),
        }
        return np.zeros(shapes.get(buffer_type, (h, w, 4)), dtype=np.float32)

    def capture(self) -> Dict[BufferType, BufferData]:
        if not self._mock:
            rep.orchestrator.step(rt_subframes=1)

        return {
            bt: BufferData(
                buffer_type=bt,
                data=self._mock_data(bt) if self._mock else self._annotators[bt].get_data(),
            )
            for bt in (self.buffers if self._mock else self._annotators)
        }

    def get_buffer(self, buffer_type: BufferType) -> BufferData:
        if not self._mock and buffer_type not in self._annotators:
            raise ValueError(f"Buffer '{buffer_type}' is not attached.")

        if not self._mock:
            rep.orchestrator.step(rt_subframes=1)

        data = self._mock_data(buffer_type) if self._mock else self._annotators[buffer_type].get_data()
        return BufferData(buffer_type=buffer_type, data=data)

    def add_buffer(self, buffer_type: BufferType):
        if buffer_type in self._annotators or (self._mock and buffer_type in self.buffers):
            return
        if not self._mock:
            self._attach(buffer_type)
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
        if self._camera:
            self._camera.set_world_pose(position=list(position))
            if target:
                self._camera.set_focus_distance(
                    float(np.linalg.norm(np.array(target) - np.array(position)))
                )

    def get_intrinsics(self) -> dict:
        if self._mock or not self._camera:
            w, h = self.resolution
            return {"focal_length": 24.0, "horizontal_aperture": 20.955, "resolution": (w, h)}
        return {
            "focal_length": self._camera.get_focal_length(),
            "horizontal_aperture": self._camera.get_horizontal_aperture(),
            "resolution": self.resolution,
        }

    def destroy(self):
        if not self._mock:
            for bt in list(self._annotators.keys()):
                self._detach(bt)
            if self._render_product:
                self._render_product.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()

    def __repr__(self) -> str:
        mode = "mock" if self._mock else "live"
        return f"SuperCamera(path={self.prim_path}, resolution={self.resolution}, buffers={[b.name for b in self.buffers]}, mode={mode})"
