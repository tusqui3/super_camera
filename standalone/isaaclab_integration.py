"""Isaac Lab integration example — IR synthesis inside a training/sim loop.

Isaac Lab (and any Isaac Sim app loop) already advances the renderer every step
via `world.step(render=True)` / `sim.render()`. In that case SuperCamera must NOT
call `rep.orchestrator.step()` itself — doing so would double-step the app. Use
`read()` / `synthesize_ir_from_render()`, which collect the already-rendered
frame without touching the orchestrator.

Pattern:
    1. Build (or reuse) a camera prim in the scene — pass its prim_path.
    2. Construct SuperCamera once with the buffers your heuristic needs.
    3. Each step, after world.step(render=True), call synthesize_ir_from_render().

Run on a machine with Isaac Lab installed:

    python standalone/isaaclab_integration.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from super.camera import SuperCamera, BufferType


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0))

    # Minimal scene so the camera has something to look at.
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/light", sim_utils.DomeLightCfg())
    cfg_cube = sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0))
    cfg_cube.func("/World/cube", cfg_cube, translation=(0.0, 0.0, 0.5))

    # SuperCamera attaches a standalone render product to the camera prim. It does
    # NOT touch the active viewport, so it is safe alongside Isaac Lab sensors.
    camera = SuperCamera(
        prim_path="/World/SuperCamera",
        resolution=(640, 480),
        buffers=[BufferType.DISTANCE_TO_OBJECT, BufferType.NORMALS],
    )
    camera.aim(position=(3.0, 0.0, 2.0), target=(0.0, 0.0, 0.5))

    sim.reset()

    for step in range(10):
        # Isaac Lab renders here — do NOT let SuperCamera step the orchestrator.
        sim.step(render=True)

        # Read the frame the app just rendered (no orchestrator.step()).
        ir = camera.synthesize_ir_from_render(mode="thermal")  # float32 (H,W) in [0,1]

        # Feed `ir` straight into an observation buffer, reward term, etc.
        # Convert to an RGB thermal image only when you want to look at it:
        #   rgb = SuperCamera.colorize(ir, "ironbow")
        print(f"step {step:02d}  ir={ir.shape} mean={float(ir.mean()):.4f}")

    camera.destroy()
    simulation_app.close()


if __name__ == "__main__":
    main()
