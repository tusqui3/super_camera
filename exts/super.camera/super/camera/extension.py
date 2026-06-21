import asyncio
import os
import numpy as np
import omni.ext
import omni.kit.app
import omni.ui as ui
import carb

from .super_camera import SuperCamera
from .buffers import SPECTRAL_BANDS, EMISSIVE

# Bands whose synthesis uses the camera-position field (active illuminators).
# Emissive bands (MWIR/LWIR) use the ambient-temperature field instead; VIS is
# passive ambient and uses neither.
_ACTIVE_BANDS = {"NIR_ACTIVE", "SWIR_ACTIVE"}
_BAND_NAMES = list(SPECTRAL_BANDS.keys())
_DEFAULT_BAND_IDX = _BAND_NAMES.index("LWIR")

try:
    import omni.kit.ui as _kit_ui
    _HAS_MENU = True
except ImportError:
    _HAS_MENU = False

try:
    import omni.kit.viewport.utility as _vpu
    _HAS_VPU = True
except ImportError:
    _HAS_VPU = False

_OUTPUT_DEFAULT = os.path.join(os.path.expanduser("~"), "super_camera_ir.png").replace("\\", "/")
_PREVIEW_W, _PREVIEW_H = 320, 180


class SuperCameraExtension(omni.ext.IExt):

    def on_startup(self, ext_id: str):
        carb.log_info("[super.camera] on_startup called")
        try:
            self._camera: SuperCamera | None = None
            self._mode_idx = _DEFAULT_BAND_IDX
            self._window: ui.Window | None = None
            self._viewport_window = None
            self._menu = None
            self._menu_entry = None
            self._tex_provider = None

            if _HAS_MENU:
                try:
                    self._menu = _kit_ui.get_editor_menu()
                    if self._menu:
                        self._menu_entry = self._menu.add_item(
                            "Window/Super Camera", self._on_menu_click, toggle=True, value=True
                        )
                except Exception as exc:
                    carb.log_warn(f"[super.camera] Menu registration failed: {exc}")

            asyncio.ensure_future(self._deferred_show())
            carb.log_info("[super.camera] on_startup complete")
        except Exception as exc:
            carb.log_error(f"[super.camera] on_startup failed: {exc}")

    async def _deferred_show(self):
        try:
            await omni.kit.app.get_app().next_update_async()
            self._show_window(True)
        except Exception as exc:
            carb.log_error(f"[super.camera] deferred show failed: {exc}")

    def _on_menu_click(self, _menu, toggled: bool):
        self._show_window(toggled)

    def _show_window(self, show: bool):
        try:
            if show:
                if self._window is None:
                    self._window = ui.Window(
                        "Super Camera", width=380, height=580, visible=True
                    )
                    self._window.set_visibility_changed_fn(self._on_visibility_changed)
                    try:
                        with self._window.frame:
                            self._build_ui()
                    except Exception as exc:
                        carb.log_error(f"[super.camera] UI build failed: {exc}")
                        with self._window.frame:
                            with ui.VStack():
                                ui.Label(f"UI error: {exc}", word_wrap=True)
                else:
                    self._window.visible = True
            elif self._window:
                self._window.visible = False
        except Exception as exc:
            carb.log_error(f"[super.camera] _show_window failed: {exc}")

    def _on_visibility_changed(self, visible: bool):
        try:
            if self._menu and self._menu_entry:
                self._menu.set_value("Window/Super Camera", visible)
        except Exception:
            pass

    def on_shutdown(self):
        try:
            if self._camera:
                self._camera.destroy()
                self._camera = None
            if self._viewport_window:
                try:
                    self._viewport_window.destroy()
                except Exception:
                    pass
                self._viewport_window = None
            if self._menu and self._menu_entry:
                try:
                    self._menu.remove_item(self._menu_entry)
                except Exception:
                    pass
            if self._window:
                self._window.destroy()
                self._window = None
        except Exception as exc:
            carb.log_error(f"[super.camera] on_shutdown failed: {exc}")

    def _build_ui(self):
        try:
            self._tex_provider = ui.ByteImageProvider()
            self._tex_provider.set_bytes_data(
                [0] * (_PREVIEW_W * _PREVIEW_H * 4), [_PREVIEW_W, _PREVIEW_H]
            )
            _has_preview = True
        except Exception as exc:
            carb.log_warn(f"[super.camera] ByteImageProvider unavailable: {exc}")
            self._tex_provider = None
            _has_preview = False

        with ui.VStack(spacing=4):

            with ui.CollapsableFrame("Camera Setup", collapsed=False):
                with ui.VStack(spacing=4, height=0):
                    with ui.HStack(height=24):
                        ui.Label("Prim Path", width=90)
                        self._prim_path = ui.StringField()
                        self._prim_path.model.set_value("/World/SuperCamera")
                    with ui.HStack(height=24):
                        ui.Label("Width", width=50)
                        self._width = ui.IntField()
                        self._width.model.set_value(1280)
                        ui.Spacer(width=8)
                        ui.Label("Height", width=50)
                        self._height = ui.IntField()
                        self._height.model.set_value(720)
                    with ui.HStack(spacing=4, height=30):
                        ui.Button("Create Camera", clicked_fn=self._on_create)
                        ui.Button("Open Viewport", clicked_fn=self._on_open_viewport)

            with ui.CollapsableFrame("Spectral Band", collapsed=False):
                with ui.VStack(spacing=4, height=0):
                    with ui.HStack(height=24):
                        ui.Label("Band", width=90)
                        self._mode_combo = ui.ComboBox(_DEFAULT_BAND_IDX, *_BAND_NAMES)
                        self._mode_combo.model.add_item_changed_fn(self._on_mode_changed)

                    self._band_desc = ui.Label("", height=0, word_wrap=True)

                    with ui.HStack(height=24):
                        self._ambient_label = ui.Label("Ambient Temp (K)", width=130)
                        self._ambient_temp = ui.FloatField()
                        self._ambient_temp.model.set_value(293.0)

                    self._cam_pos_label = ui.Label(
                        "Active bands: the light source is located at the camera "
                        "eye (coaxial illumination).",
                        height=0,
                        word_wrap=True,
                    )

            if _has_preview:
                with ui.CollapsableFrame("IR Preview", collapsed=False):
                    try:
                        ui.ImageWithProvider(
                            self._tex_provider,
                            width=ui.Fraction(1),
                            height=_PREVIEW_H,
                        )
                    except Exception as exc:
                        carb.log_warn(f"[super.camera] ImageWithProvider failed: {exc}")
                        ui.Label("Preview unavailable.", height=20)

            with ui.CollapsableFrame("Output", collapsed=False):
                with ui.HStack(height=24):
                    ui.Label("Save Path", width=70)
                    self._output_path = ui.StringField()
                    self._output_path.model.set_value(_OUTPUT_DEFAULT)

            ui.Separator(height=4)
            self._status_label = ui.Label("Ready.", height=24, word_wrap=True)

            with ui.HStack(spacing=4, height=30):
                ui.Button("Capture IR Frame", clicked_fn=self._on_capture)
                ui.Button("Reset Camera", width=110, clicked_fn=self._on_reset)

        self._on_mode_changed(self._mode_combo.model, None)

    def _get_params(self):
        prim_path = self._prim_path.model.get_value_as_string()
        w = self._width.model.get_value_as_int()
        h = self._height.model.get_value_as_int()
        return prim_path, (w, h)

    def _ensure_camera(self):
        prim_path, resolution = self._get_params()
        if (self._camera is None
                or prim_path != self._camera.prim_path
                or resolution != self._camera.resolution):
            if self._camera:
                self._camera.destroy()
            self._camera = SuperCamera(prim_path=prim_path, resolution=resolution)

    def _on_create(self):
        try:
            self._ensure_camera()
            prim_path, _ = self._get_params()
            self._status_label.text = f"Camera created at {prim_path}"
        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            carb.log_error(f"[super.camera] Create failed: {exc}")

    def _on_open_viewport(self):
        if not _HAS_VPU:
            self._status_label.text = "omni.kit.viewport.utility not available."
            carb.log_warn("[super.camera] omni.kit.viewport.utility not found")
            return
        try:
            self._ensure_camera()
            prim_path, (w, h) = self._get_params()
            if self._viewport_window is None:
                self._viewport_window = _vpu.create_viewport_window(
                    "Super Camera Preview",
                    width=w // 2,
                    height=h // 2,
                )
            self._viewport_window.visible = True
            asyncio.ensure_future(self._assign_viewport_camera(prim_path))
            self._status_label.text = "Viewport opened."
        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            carb.log_error(f"[super.camera] Viewport open failed: {exc}")

    async def _assign_viewport_camera(self, prim_path):
        app = omni.kit.app.get_app()
        for _ in range(20):
            await app.next_update_async()
            window = self._viewport_window
            if window is None:
                return
            vp_api = getattr(window, "viewport_api", None)
            if vp_api is None:
                continue
            try:
                try:
                    vp_api.camera_path = prim_path
                except AttributeError:
                    vp_api.set_active_camera(prim_path)
                return
            except Exception:
                continue
        carb.log_warn(
            f"[super.camera] could not set viewport camera to {prim_path}"
        )

    def _on_mode_changed(self, model, _item):
        try:
            idx = model.get_item_value_model().get_value_as_int()
            self._mode_idx = idx if 0 <= idx < len(_BAND_NAMES) else _DEFAULT_BAND_IDX
            band = _BAND_NAMES[self._mode_idx]
            spec = SPECTRAL_BANDS[band]
            is_emissive = spec.reflective_vs_emissive == EMISSIVE
            is_active = band in _ACTIVE_BANDS

            self._band_desc.text = (
                f"{int(spec.wavelength_min_nm)}–{int(spec.wavelength_max_nm)} nm · "
                f"{spec.reflective_vs_emissive}\n{spec.description}"
            )
            # Ambient temp drives the emissive thermal bands; camera position
            # drives the active reflective bands; VIS uses neither.
            self._ambient_label.enabled = is_emissive
            self._ambient_temp.enabled = is_emissive
            self._cam_pos_label.enabled = is_active
        except Exception as exc:
            carb.log_warn(f"[super.camera] mode change failed: {exc}")

    def _on_capture(self):
        self._status_label.text = "Capturing…"
        asyncio.ensure_future(self._capture_async())

    async def _capture_async(self):
        try:
            prim_path, (w, h) = self._get_params()
            band = _BAND_NAMES[self._mode_idx]
            ambient_temp = self._ambient_temp.model.get_value_as_float()
            output_path = os.path.expanduser(os.path.expandvars(
                self._output_path.model.get_value_as_string()
            ))

            self._ensure_camera()

            ir = await self._camera.synthesize_ir_async(
                mode=band,
                ambient_temp=ambient_temp,
            )

            disp = np.clip(ir, 0.0, 1.0)
            color = SuperCamera.colorize(disp, colormap="ironbow")
            if self._tex_provider is not None:
                self._update_preview(color, w, h)
            self._save_ir(color, output_path)
            self._status_label.text = f"Saved → {output_path}"
            carb.log_info(f"[super.camera] IR frame saved to {output_path}")

        except Exception as exc:
            self._status_label.text = f"Error: {exc}"
            carb.log_error(f"[super.camera] Capture failed: {exc}")

    def _update_preview(self, rgb: np.ndarray, orig_w: int, orig_h: int):
        try:
            step = max(1, min(orig_h // _PREVIEW_H, orig_w // _PREVIEW_W))
            small = rgb[::step, ::step][:_PREVIEW_H, :_PREVIEW_W]
            ph, pw = small.shape[:2]
            alpha = np.full((ph, pw, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([small, alpha], axis=-1)
            self._tex_provider.set_bytes_data(rgba.flatten().tolist(), [pw, ph])
        except Exception as exc:
            carb.log_warn(f"[super.camera] preview update failed: {exc}")

    def _on_reset(self):
        if self._camera:
            self._camera.destroy()
            self._camera = None
        if self._viewport_window:
            try:
                self._viewport_window.destroy()
            except Exception:
                pass
            self._viewport_window = None
        self._status_label.text = "Camera reset."

    def _save_ir(self, rgb: np.ndarray, path: str):
        img = np.ascontiguousarray(rgb.astype(np.uint8))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            from PIL import Image
            Image.fromarray(img, mode="RGB").save(path)
        except ImportError:
            ppm_path = os.path.splitext(path)[0] + ".ppm"
            h, w = img.shape[:2]
            with open(ppm_path, "wb") as f:
                f.write(f"P6\n{w} {h}\n255\n".encode())
                f.write(img.tobytes())
            self._status_label.text = f"Pillow unavailable — saved PPM → {ppm_path}"
