import omni.ext
import omni.kit.ui

from .super_camera import SuperCamera
from .buffers import BufferType


class SuperCameraExtension(omni.ext.IExt):

    def on_startup(self, ext_id: str):
        print(f"[super.camera] Starting up — ext_id: {ext_id}")
        self._camera: SuperCamera | None = None

    def on_shutdown(self):
        print("[super.camera] Shutting down")
        if self._camera:
            self._camera.destroy()
            self._camera = None
