"""Standalone Isaac Sim example — capture buffers and synthesize IR imagery.

Run on a machine with Isaac Sim installed:

    python standalone/example.py

Produces three PNGs in the working directory:
    depth_synthetic.png   jet-colormapped distance image
    ir_thermal.png        thermal-mode IR, ironbow (thermal-camera) palette
    ir_active_nir.png     active-NIR-mode IR, ironbow palette
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
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

# 2) Thermal-mode IR → ironbow (thermal-camera) palette.
#    synthesize_ir returns a float32 (H,W) map in [0,1]; colorize() maps it to
#    a uint8 (H,W,3) RGB image for display. Keep the float map for training.
ir_thermal = camera.synthesize_ir(mode="thermal")
Image.fromarray(SuperCamera.colorize(ir_thermal, "ironbow")).save("ir_thermal.png")
print(f"Thermal IR saved: {ir_thermal.shape} {ir_thermal.dtype}")

# 3) Active-NIR-mode IR → ironbow palette.
ir_nir = camera.synthesize_ir(mode="active_nir", camera_pos=np.array([0.0, -5.0, 2.0]))
Image.fromarray(SuperCamera.colorize(ir_nir, "ironbow")).save("ir_active_nir.png")
print(f"Active-NIR IR saved: {ir_nir.shape} {ir_nir.dtype}")

camera.destroy()
app.close()
