import mujoco_py
import numpy as np
import time

def run_ant_simulation():
    # Load the model
    model = mujoco_py.load_model_from_path("ant_terrain.xml")
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    
    # Simulation loop
    while True:
        # Simple random control for the ant
        sim.data.ctrl[:] = 0.2 * np.random.randn(sim.model.nu)
        
        # Step the simulation and render
        sim.step()
        viewer.render()
        
        # Add a small delay to slow down the simulation
        time.sleep(0.01)

# Run the simulation
if __name__ == "__main__":
    run_ant_simulation()
