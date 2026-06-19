"""Mock-mode example — logic-only, numpy only, no Isaac Sim needed.

    python standalone/example_mock.py

All buffers return zero-filled arrays of the correct shape/dtype, so this
exercises the synthesis + colormap code paths without an Omniverse install.
"""

from super.camera import SuperCamera, BufferType

camera = SuperCamera(
    prim_path="/World/SuperCamera",
    resolution=(1280, 720),
    buffers=[BufferType.DISTANCE_TO_OBJECT],
    mock=True,
)

print(camera)
print("Intrinsics:", camera.get_intrinsics())

frame = camera.capture()
for buf_type, buf_data in frame.items():
    print(buf_data)

img = camera.synthesize(max_distance=20.0, colormap="jet")
print(f"Depth (jet):       shape={img.shape}, dtype={img.dtype}")

img_thermal = camera.synthesize(max_distance=20.0, colormap="ironbow")
print(f"Depth (ironbow):   shape={img_thermal.shape}, dtype={img_thermal.dtype}")

# Exercise every spectral band's synthesis path (mock buffers are zero-filled,
# so values are uniform — this checks shapes/dtypes, not realism).
for band in ("VIS", "NIR_ACTIVE", "SWIR_ACTIVE", "MWIR", "LWIR"):
    ir = camera.synthesize_ir(band)
    print(f"IR {band:12s} (float): shape={ir.shape}, dtype={ir.dtype}, "
          f"range=[{ir.min():.3f},{ir.max():.3f}]")

# Deprecated aliases still resolve (thermal → LWIR, active_nir → NIR_ACTIVE).
ir = camera.synthesize_ir(mode="thermal")
ir_rgb = SuperCamera.colorize(ir, "ironbow")
print(f"IR thermal→LWIR (rgb): shape={ir_rgb.shape}, dtype={ir_rgb.dtype}")
