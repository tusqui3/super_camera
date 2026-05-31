# Super Camera 🎥

A small Isaac Sim / Omniverse extension I put together to make grabbing render buffer data less painful. It wraps the Replicator AOV pipeline so you don't have to wire up render products and annotators by hand every single time.

```python
from super.camera import SuperCamera, BufferType

camera = SuperCamera(
    prim_path="/World/SuperCamera",
    resolution=(1280, 720),
    buffers=[BufferType.RGB, BufferType.DEPTH, BufferType.SEMANTIC],
)

frame = camera.capture()
depth = frame[BufferType.DEPTH].data   # numpy array, shape (H, W, 1)
rgb   = frame[BufferType.RGB].data     # numpy array, shape (H, W, 4)
```

## Supported buffers

| BufferType | Annotator | Output |
|---|---|---|
| `RGB` | `rgb` | `(H, W, 4)` uint8 |
| `DEPTH` | `distance_to_image_plane` | `(H, W, 1)` float32 |
| `NORMALS` | `normals` | `(H, W, 4)` float32 |
| `SEMANTIC` | `semantic_segmentation` | `(H, W, 4)` uint32 |
| `INSTANCE` | `instance_segmentation` | `(H, W, 4)` uint32 |
| `MOTION_VECTORS` | `motion_vectors` | `(H, W, 4)` float32 |
| `OCCLUSION` | `occlusion` | `(H, W, 1)` float32 |
| `ALBEDO` | `albedo` | `(H, W, 4)` float32 |

## Installation

```bash
git clone https://github.com/yourusername/super_camera.git
```

In Omniverse / Isaac Sim go to **Window → Extensions → ⚙ → Add path** and point it at:

```
/path/to/super_camera/exts
```

Search for **Super Camera** and enable it.

## Usage

### Standalone / headless

```python
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from super.camera import SuperCamera, BufferType

with SuperCamera("/World/Cam", buffers=[BufferType.RGB, BufferType.DEPTH]) as cam:
    cam.set_pose(position=(0, -5, 2), target=(0, 0, 0))
    frame = cam.capture()
    print(frame[BufferType.DEPTH])

app.close()
```

### Add / remove buffers at runtime

```python
camera.add_buffer(BufferType.NORMALS)
camera.remove_buffer(BufferType.OCCLUSION)
```

### Single buffer

```python
depth = camera.get_buffer(BufferType.DEPTH)
print(depth.data.min(), depth.data.max())
```

### Camera intrinsics

```python
print(camera.get_intrinsics())
# {'focal_length': 24.0, 'horizontal_aperture': 20.955, 'resolution': (1280, 720)}
```

### Mock mode

Handy for working on things locally without a full Omniverse install, and for CI. Just returns zero-filled arrays with the right shapes.

```python
camera = SuperCamera(mock=True, buffers=[BufferType.RGB, BufferType.DEPTH])
frame = camera.capture()
```

Or run it directly:

```bash
python standalone/example_mock.py
```

## Project structure

```
super_camera/
├── exts/
│   └── super.camera/
│       ├── config/
│       │   └── extension.toml
│       └── super/camera/
│           ├── __init__.py
│           ├── extension.py       # Omniverse extension lifecycle
│           ├── super_camera.py    # SuperCamera class
│           └── buffers.py        # BufferType enum + BufferData
├── standalone/
│   ├── example.py                 # Full Isaac Sim example
│   └── example_mock.py            # No Omniverse needed
└── README.md
```

## Requirements

- NVIDIA Omniverse or Isaac Sim 4.x
- Python 3.10+
- `numpy`

Mock mode only needs `numpy`.