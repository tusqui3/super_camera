# Image placeholders

Drop screenshots / renders here to populate the main README gallery. Expected files:

| File | What it should show |
|---|---|
| `hero_thermal.png` | A thermal-mode IR render with the ironbow palette (top banner) |
| `rgb.png` | The plain RGB render of the same scene |
| `depth_jet.png` | Distance image via `synthesize(colormap="jet")` |
| `ir_thermal.png` | Thermal-mode IR, ironbow palette |
| `ir_active_nir.png` | Active-NIR-mode IR, ironbow palette |
| `gui_panel.png` | The Super Camera extension GUI panel |

Generate `depth_jet.png`, `ir_thermal.png`, and `ir_active_nir.png` directly with
`standalone/example.py`.
