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

ir = camera.synthesize_ir(mode="thermal")
print(f"IR thermal (float):shape={ir.shape}, dtype={ir.dtype}, range=[{ir.min():.3f},{ir.max():.3f}]")

ir_rgb = SuperCamera.colorize(ir, "ironbow")
print(f"IR thermal (rgb):  shape={ir_rgb.shape}, dtype={ir_rgb.dtype}")
