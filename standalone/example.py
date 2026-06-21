"""Standalone Isaac Sim example — capture buffers and synthesize IR imagery.

Run on a machine with Isaac Sim installed:

    python standalone/example.py

Produces, in the working directory:
    depth_synthetic.png   jet-colormapped distance image
    ir_VIS.png            visible reflectance         (400–700 nm, reflective)
    ir_NIR_ACTIVE.png     active near-infrared        (700–1000 nm, reflective)
    ir_SWIR_ACTIVE.png    active short-wave infrared   (1000–2500 nm, reflective)
    ir_MWIR.png           mid-wave thermal emission    (3000–5000 nm, emissive)
    ir_LWIR.png           long-wave thermal emission   (8000–14000 nm, emissive)
all IR frames rendered with the ironbow (thermal-camera) palette.
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from PIL import Image
from super.camera import SuperCamera, BufferType

camera = SuperCamera(
    prim_path="/World/SuperCamera",
    resolution=(1280, 720),
    buffers=[
        BufferType.DISTANCE_TO_OBJECT,
        # BufferType.DEPTH,
        # BufferType.RGB,
        # BufferType.NORMALS,
        # BufferType.DIFFUSE_ALBEDO,
        # BufferType.SPECULAR_ALBEDO,
        # BufferType.ROUGHNESS,
        # BufferType.EMISSIVE,
        # BufferType.MOTION_VECTORS,
        # BufferType.SEMANTIC,
        # BufferType.INSTANCE,
        # BufferType.INSTANCE_ID,
        # BufferType.OCCLUSION,
        # BufferType.BBOX_2D_TIGHT,
        # BufferType.BBOX_2D_LOOSE,
        # BufferType.BBOX_3D,
        # BufferType.CAMERA_PARAMS,
        # BufferType.POINTCLOUD,
        # BufferType.SKELETON,
    ],
)

print(camera)
print("Intrinsics:", camera.get_intrinsics())

# Aim the camera at the scene origin (Z-up world). Without an aim/pose the prim
# sits at the origin looking down -Z and typically sees empty space (all inf).
camera.aim(position=(0.0, -5.0, 2.0), target=(0.0, 0.0, 0.0))

frame = camera.capture()
for buf_type, buf_data in frame.items():
    print(buf_data)

# 1) Depth → jet colormap
img = camera.synthesize(max_distance=20.0, colormap="jet")
Image.fromarray(img).save("depth_synthetic.png")
print(f"Depth image saved: {img.shape} {img.dtype}")

# 2) One IR frame per spectral band → ironbow (thermal-camera) palette.
#    synthesize_ir returns a float32 (H,W) map in [0,1]; colorize() maps it to a
#    uint8 (H,W,3) RGB image for display. Keep the float map for training.
#    The active bands (NIR/SWIR) assume a coaxial illuminator at the camera eye;
#    the emissive bands (MWIR/LWIR) read ambient_temp instead.
for band in ("VIS", "NIR_ACTIVE", "SWIR_ACTIVE", "MWIR", "LWIR"):
    ir = camera.synthesize_ir(band, ambient_temp=293.0)
    Image.fromarray(SuperCamera.colorize(ir, "ironbow")).save(f"ir_{band}.png")
    print(f"{band:12s} IR saved: {ir.shape} {ir.dtype}")

camera.destroy()
app.close()
