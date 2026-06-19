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
    # Diffuse PBR albedo color (RTX AOV)                      (H,W,4) float
    DIFFUSE_ALBEDO = "DiffuseAlbedo"
    # Specular PBR color (RTX AOV)                            (H,W,4) float
    SPECULAR_ALBEDO = "SpecularAlbedo"
    # Surface roughness (RTX AOV)                             (H,W[,C]) float
    ROUGHNESS = "Roughness"
    # Emission color + foreground mask (RTX AOV)              (H,W,4) float
    EMISSIVE = "EmissionAndForegroundMask"

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

# Some annotators are exposed under different strings across Isaac Sim / Kit
# versions. The primary name is BufferType.value (ANNOTATOR_MAP); these are tried
# in order as fallbacks when the primary isn't registered — e.g. the PBR material
# AOVs are CamelCase on Kit 107.x (DiffuseAlbedo) but were lowercase on older
# builds (diffuse_albedo). _attach() walks primary → fallbacks and uses the first
# that the AnnotatorRegistry knows.
ANNOTATOR_FALLBACKS: dict[BufferType, list[str]] = {
    BufferType.DIFFUSE_ALBEDO:  ["diffuse_albedo"],
    BufferType.SPECULAR_ALBEDO: ["specular_albedo"],
    BufferType.ROUGHNESS:       ["roughness"],
    BufferType.EMISSIVE:        ["emissive", "Emission"],
}

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


# ── Spectral bands ───────────────────────────────────────────────────────────
# Source-of-truth definitions for the synthetic IR spectral bands. Each band is
# identified by its wavelength range and classified as primarily REFLECTIVE
# (the sensor sees illumination bouncing off surfaces) or EMISSIVE (the sensor
# sees thermal radiation the surface itself emits). SuperCamera.synthesize_ir()
# selects a band by name and dispatches to its per-band synthesis model; the GUI
# and the docs read these same definitions, so band facts live in exactly one
# place.

REFLECTIVE = "reflective"
EMISSIVE = "emissive"


@dataclass(frozen=True)
class SpectralBand:
    name: str                       # canonical band name (the mode= selector)
    wavelength_min_nm: float        # lower edge of the band, nanometres
    wavelength_max_nm: float        # upper edge of the band, nanometres
    reflective_vs_emissive: str     # REFLECTIVE or EMISSIVE
    description: str                 # sensor behaviour + dominant scene physics


SPECTRAL_BANDS: dict[str, SpectralBand] = {
    "VIS": SpectralBand(
        name="VIS",
        wavelength_min_nm=400.0,
        wavelength_max_nm=700.0,
        reflective_vs_emissive=REFLECTIVE,
        description=(
            "Visible reflectance. Passive ambient illumination reflected off "
            "surfaces — diffuse + specular PBR response, no active illuminator "
            "and no distance falloff."
        ),
    ),
    "NIR_ACTIVE": SpectralBand(
        name="NIR_ACTIVE",
        wavelength_min_nm=700.0,
        wavelength_max_nm=1000.0,
        reflective_vs_emissive=REFLECTIVE,
        description=(
            "Active near-infrared. Coaxial illuminator (light source = camera); "
            "reflected energy with strong specular highlights and inverse-square "
            "distance falloff. Smooth surfaces give concentrated returns."
        ),
    ),
    "SWIR_ACTIVE": SpectralBand(
        name="SWIR_ACTIVE",
        wavelength_min_nm=1000.0,
        wavelength_max_nm=2500.0,
        reflective_vs_emissive=REFLECTIVE,
        description=(
            "Active short-wave infrared. Like NIR_ACTIVE but largely colour-"
            "blind: grayscale reflectance with greater emphasis on material and "
            "roughness, plus specular response and inverse-square falloff."
        ),
    ),
    "MWIR": SpectralBand(
        name="MWIR",
        wavelength_min_nm=3000.0,
        wavelength_max_nm=5000.0,
        reflective_vs_emissive=EMISSIVE,
        description=(
            "Mid-wave infrared thermal emission. Intensity from emissivity and "
            "estimated temperature with a steep (T^4-style) response — strongly "
            "biased toward hot objects (emissive materials, motion-derived heat)."
        ),
    ),
    "LWIR": SpectralBand(
        name="LWIR",
        wavelength_min_nm=8000.0,
        wavelength_max_nm=14000.0,
        reflective_vs_emissive=EMISSIVE,
        description=(
            "Long-wave infrared thermal emission. Ambient-temperature band — "
            "every surface emits; brightness is set by emissivity and ambient "
            "temperature, geometry has only weak influence, no distance falloff."
        ),
    ),
}

# Deprecated mode aliases kept for backward compatibility. They map onto the
# canonical band names above; SuperCamera emits a one-time deprecation notice
# the first time each alias is used.
DEPRECATED_BAND_ALIASES: dict[str, str] = {
    "thermal": "LWIR",
    "active_nir": "NIR_ACTIVE",
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
