from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class BufferType(str, Enum):
    RGB = "rgb"
    DEPTH = "distance_to_image_plane"
    NORMALS = "normals"
    SEMANTIC = "semantic_segmentation"
    INSTANCE = "instance_segmentation"
    MOTION_VECTORS = "motion_vectors"
    OCCLUSION = "occlusion"
    ALBEDO = "albedo"


ANNOTATOR_MAP: dict[BufferType, str] = {
    BufferType.RGB: "rgb",
    BufferType.DEPTH: "distance_to_image_plane",
    BufferType.NORMALS: "normals",
    BufferType.SEMANTIC: "semantic_segmentation",
    BufferType.INSTANCE: "instance_segmentation",
    BufferType.MOTION_VECTORS: "motion_vectors",
    BufferType.OCCLUSION: "occlusion",
    BufferType.ALBEDO: "albedo",
}


@dataclass
class BufferData:
    buffer_type: BufferType
    data: np.ndarray
    metadata: Optional[dict] = None

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self) -> str:
        return f"BufferData(type={self.buffer_type.name}, shape={self.shape}, dtype={self.dtype})"
