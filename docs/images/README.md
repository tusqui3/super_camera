# Image placeholders

Drop screenshots / renders here to populate the main README gallery. Expected files:

| File | What it should show |
|---|---|
| `hero_thermal.png` | An LWIR thermal IR render with the ironbow palette (top banner) |
| `rgb.png` | The plain RGB render of the same scene |
| `depth_jet.png` | Distance image via `synthesize(colormap="jet")` |
| `ir_VIS.png` | Visible reflectance (400–700 nm, reflective), ironbow palette |
| `ir_NIR_ACTIVE.png` | Active near-infrared (700–1000 nm, reflective), ironbow palette |
| `ir_SWIR_ACTIVE.png` | Active short-wave infrared (1000–2500 nm, reflective), ironbow palette |
| `ir_MWIR.png` | Mid-wave thermal emission (3000–5000 nm, emissive), ironbow palette |
| `ir_LWIR.png` | Long-wave thermal emission (8000–14000 nm, emissive), ironbow palette |
| `gui_panel.png` | The Super Camera extension GUI panel |

Generate `depth_jet.png` and the five `ir_<BAND>.png` frames directly with
`standalone/example.py`.
