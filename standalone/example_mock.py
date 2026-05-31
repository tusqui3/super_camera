from super.camera import SuperCamera, BufferType

camera = SuperCamera(
    prim_path="/World/SuperCamera",
    resolution=(1280, 720),
    buffers=[BufferType.RGB, BufferType.DEPTH, BufferType.NORMALS],
    mock=True,
)

print(camera)
print("Intrinsics:", camera.get_intrinsics())

frame = camera.capture()
for buf_type, buf_data in frame.items():
    print(buf_data)
