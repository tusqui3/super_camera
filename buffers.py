from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np


class BufferType(str, Enum):
    # ── Active ─────────────────────────────────────────────────────────────────
    # Euclidean distance from camera origin to each surface point (H,W) float32
    DISTANCE_TO_OBJECT = "distance_to_camera"

    # ── Depth / Geometry ───────────────────────────────────────────────────────
    # Orthogonal (perpendicular) depth to image plane         (H,W) float32
    DEPTH = "distance_to_image_plane"
    # World-space surface normals, XYZ + padding              (H,W,4) float32
    NORMALS = "normals"

    # ── Color / Appearance ─────────────────────────────────────────────────────
    # RGBA color image                                        (H,W,4) uint8
    RGB = "rgb"
    # Diffuse PBR albedo color                                (H,W,4) float32
    DIFFUSE_ALBEDO = "diffuse_albedo"
    # Specular PBR color                                      (H,W,4) float32
    SPECULAR_ALBEDO = "specular_albedo"
    # Surface roughness scalar                                (H,W) float32
    ROUGHNESS = "roughness"
    # Emissive color                                          (H,W,4) float32
    EMISSIVE = "emissive"

    # ── Motion ─────────────────────────────────────────────────────────────────
    # 2-D screen-space motion vectors                         (H,W,4) float32
    MOTION_VECTORS = "motion_vectors"

    # ── Segmentation (get_data() returns dict with 'data' ndarray + 'info') ───
    # Per-pixel semantic class label                          dict → data (H,W) uint32
    SEMANTIC = "semantic_segmentation"
    # Per-instance hierarchical segmentation                  dict → data (H,W) uint32
    INSTANCE = "instance_segmentation"
    # Per-leaf-prim instance ID segmentation                  dict → data (H,W) uint32
    INSTANCE_ID = "instance_id_segmentation"

    # ── Scene Analysis (get_data() returns structured dicts, NOT pixel buffers) ─
    # Per-instance occlusion ratio [0=visible, 1=occluded]    dict
    OCCLUSION = "occlusion"
    # Tight 2-D axis-aligned bounding boxes                   dict → data structured array
    BBOX_2D_TIGHT = "bounding_box_2d_tight"
    # Loose 2-D axis-aligned bounding boxes                   dict → data structured array
    BBOX_2D_LOOSE = "bounding_box_2d_loose"
    # 3-D oriented bounding boxes + world pose                dict → data structured array
    BBOX_3D = "bounding_box_3d"
    # Camera intrinsics and extrinsics                        dict
    CAMERA_PARAMS = "camera_params"
    # 3-D point cloud (world-space)                           dict → data (N,3) float32
    POINTCLOUD = "pointcloud"
    # Character skeleton joint positions                      dict
    SKELETON = "skeleton_data"


ANNOTATOR_MAP: dict[BufferType, str] = {bt: bt.value for bt in BufferType}

STRUCTURED_BUFFERS: frozenset[BufferType] = frozenset({
    BufferType.SEMANTIC,
    BufferType.INSTANCE,
    BufferType.INSTANCE_ID,
    BufferType.OCCLUSION,
    BufferType.BBOX_2D_TIGHT,
    BufferType.BBOX_2D_LOOSE,
    BufferType.BBOX_3D,
    BufferType.CAMERA_PARAMS,
    BufferType.POINTCLOUD,
    BufferType.SKELETON,
})

MOCK_SHAPES: dict[BufferType, tuple] = {
    BufferType.DISTANCE_TO_OBJECT: (),       # scalar shape; actual = (H,W)
    BufferType.DEPTH:              (),
    BufferType.NORMALS:            (4,),
    BufferType.RGB:                (4,),
    BufferType.DIFFUSE_ALBEDO:     (4,),
    BufferType.SPECULAR_ALBEDO:    (4,),
    BufferType.ROUGHNESS:          (),
    BufferType.EMISSIVE:           (4,),
    BufferType.MOTION_VECTORS:     (4,),
}


@dataclass
class BufferData:
    buffer_type: BufferType
    data: Any
    metadata: Optional[dict] = field(default=None)

    @property
    def is_pixel_buffer(self) -> bool:
        return isinstance(self.data, np.ndarray) and self.data.ndim >= 2

    @property
    def shape(self) -> tuple:
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        return ()

    @property
    def dtype(self):
        if isinstance(self.data, np.ndarray):
            return self.data.dtype
        return type(self.data)

    def __repr__(self) -> str:
        if self.is_pixel_buffer:
            return f"BufferData(type={self.buffer_type.name}, shape={self.shape}, dtype={self.dtype})"
        return f"BufferData(type={self.buffer_type.name}, structured=True)"
