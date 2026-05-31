from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
from super.camera import SuperCamera, BufferType

camera = SuperCamera(
    prim_path="/World/SuperCamera",
    resolution=(1280, 720),
    buffers=[
        BufferType.RGB,
        BufferType.DEPTH,
        BufferType.NORMALS,
        BufferType.SEMANTIC,
    ],
)

print(camera)
print("Intrinsics:", camera.get_intrinsics())

camera.set_pose(position=(0.0, -5.0, 2.0), target=(0.0, 0.0, 0.0))

frame = camera.capture()
for buf_type, buf_data in frame.items():
    print(buf_data)

camera.add_buffer(BufferType.INSTANCE)
instance = camera.get_buffer(BufferType.INSTANCE)
print("Instance segmentation shape:", instance.shape)

camera.remove_buffer(BufferType.NORMALS)

with SuperCamera(prim_path="/World/DepthOnly", buffers=[BufferType.DEPTH]) as depth_cam:
    data = depth_cam.get_buffer(BufferType.DEPTH)
    arr = data.data
    print(f"Depth — min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}")

camera.destroy()
app.close()
