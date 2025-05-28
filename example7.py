from jax import numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os

from jhsfm.hsfm import step
from jhsfm.utils import *

# Hyperparameters
grid_cell_size = 1
grid_distance_threshold = 1
dt = 0.01
end_time = 15
num_obstacles = 1400
obstacle_min_size = 0.3
obstacle_max_size = 1.0
# Prepare simulation data
steps = int(end_time/dt)
n_simulations = 1

# Humans state - example with 43 humans in a line
humans_state = np.array([
    [7.,0.,0.,0.,jnp.pi,0.],
    [6.8,0.8,0.,0.,jnp.pi,0.],
    [6.8,-0.8,0.,0.,jnp.pi,0.],
    [6.5,1.5,0.,0.,jnp.pi,0.],
    [6.5,-1.5,0.,0.,jnp.pi,0.],
    [6.2,2.2,0.,0.,jnp.pi,0.],
    [6.2,-2.2,0.,0.,jnp.pi,0.],
    [6.0,3.0,0.,0.,jnp.pi,0.],
    [6.0,-3.0,0.,0.,jnp.pi,0.],
    [5.8,3.8,0.,0.,jnp.pi,0.],
    [5.8,-3.8,0.,0.,jnp.pi,0.],
    [5.5,4.5,0.,0.,jnp.pi,0.],
    [5.5,-4.5,0.,0.,jnp.pi,0.],
    [5.2,5.2,0.,0.,jnp.pi,0.],
    [5.2,-5.2,0.,0.,jnp.pi,0.],
    [5.0,6.0,0.,0.,jnp.pi,0.],
    [5.0,-6.0,0.,0.,jnp.pi,0.],
    [4.8,6.8,0.,0.,jnp.pi,0.],
    [4.8,-6.8,0.,0.,jnp.pi,0.],
    [4.5,7.5,0.,0.,jnp.pi,0.],
    [4.5,-7.5,0.,0.,jnp.pi,0.],
    [4.2,8.2,0.,0.,jnp.pi,0.],
    [4.2,-8.2,0.,0.,jnp.pi,0.],
    [4.0,9.0,0.,0.,jnp.pi,0.],
    [4.0,-9.0,0.,0.,jnp.pi,0.],
    [3.8,9.8,0.,0.,jnp.pi,0.],
    [3.8,-9.8,0.,0.,jnp.pi,0.],
    [3.5,10.5,0.,0.,jnp.pi,0.],
    [3.5,-10.5,0.,0.,jnp.pi,0.],
    [3.2,11.2,0.,0.,jnp.pi,0.],
    [3.2,-11.2,0.,0.,jnp.pi,0.],
    [3.0,12.0,0.,0.,jnp.pi,0.],
    [3.0,-12.0,0.,0.,jnp.pi,0.],
    [2.8,12.8,0.,0.,jnp.pi,0.],
    [2.8,-12.8,0.,0.,jnp.pi,0.],
    [2.5,13.5,0.,0.,jnp.pi,0.],
    [2.5,-13.5,0.,0.,jnp.pi,0.],
    [2.2,14.2,0.,0.,jnp.pi,0.],
    [2.2,-14.2,0.,0.,jnp.pi,0.],
    [2.0,15.0,0.,0.,jnp.pi,0.],
    [2.0,-15.0,0.,0.,jnp.pi,0.],
    [1.8,15.8,0.,0.,jnp.pi,0.],
    [1.8,-15.8,0.,0.,jnp.pi,0.],
])
# Static obstacles - example adding some padding edges as dimensions should be equal for the static_obstacles array but obstacles may have different number of edges and could be dfferentiated for each human (for optimization)
# Generate random 4-edge rectangular obstacles in x ∈ (-30, 0), y ∈ (-30, 30)
rng = np.random.default_rng(42)
static_obstacles = []
for _ in range(num_obstacles):
    cx = rng.uniform(-30, 0)
    cy = rng.uniform(-30, 30)
    w = rng.uniform(obstacle_min_size, obstacle_max_size)
    h = rng.uniform(obstacle_min_size, obstacle_max_size)
    # Rectangle corners (clockwise)
    p1 = [cx - w/2, cy - h/2]
    p2 = [cx + w/2, cy - h/2]
    p3 = [cx + w/2, cy + h/2]
    p4 = [cx - w/2, cy + h/2]
    # 4 edges as pairs of points
    static_obstacles.append([
        [p1, p2],
        [p2, p3],
        [p3, p4],
        [p4, p1]
    ])
static_obstacles = jnp.array(static_obstacles)
static_obstacles_per_cell, new_static_obstacles, grid_coords = grid_cell_obstacle_occupancy(static_obstacles, grid_cell_size, grid_distance_threshold)
print(f"Number of humans: {len(humans_state)}")
print(f"Number of static obstacles: {len(static_obstacles)}")
print(f"Number of grid cells: {static_obstacles_per_cell.shape[0]} x {static_obstacles_per_cell.shape[1]} = {static_obstacles_per_cell.shape[0]*static_obstacles_per_cell.shape[1]}")
print(f"Max n of obstacles per grid cell: {static_obstacles_per_cell.shape[2]}")
print(f"Grid cell size: {grid_cell_size}")
print(f"Grid distance threshold: {grid_distance_threshold}")

# Initial conditions
humans_goal = np.zeros((len(humans_state), 2))
for i in range(len(humans_state)):
    # Goal: (gx, gy)
    humans_goal[i,0] = -7
    humans_goal[i,1] = 0.
humans_state = jnp.array(humans_state)
initial_humans_state = jnp.copy(humans_state)
humans_parameters = get_standard_humans_parameters(len(humans_state))
humans_goal = jnp.array(humans_goal)

# Dummy step - Warm-up (we first compile the JIT functions to avoid counting compilation time later)
dummy_static_obstacles = jnp.stack([static_obstacles for _ in range(len(humans_state))])
_ = step(humans_state, humans_goal, humans_parameters, dummy_static_obstacles, dt)
test_obstacles = filter_obstacles(humans_state, new_static_obstacles, static_obstacles_per_cell, grid_coords, grid_cell_size)
_ = step(humans_state, humans_goal, humans_parameters, test_obstacles, dt)
print(f"\nAvailable devices: {jax.devices()}\n")

# Simulation NOT FILTERING OBSTACLES
print(f"Starting simulation NOT FILTERING OBSTACLES... - N° simulations {n_simulations} - Simulation time: {steps*dt} seconds")
start_time = time.time()
for _ in range(n_simulations):
    humans_state = jnp.copy(initial_humans_state)
    for _ in range(steps):
        humans_state = step(humans_state, humans_goal, humans_parameters, dummy_static_obstacles, dt)
        humans_state.block_until_ready() # Wait for the computation to finish
end_time = time.time()
print("Simulations done! Average execution time per simulation: ", (end_time - start_time)/n_simulations)

# Simulation FILTERING OBSTACLES
humans_state = jnp.copy(initial_humans_state)
print(f"\nStarting simulation FILTERING OBSTACLES... - N° simulations {n_simulations} - Simulation time: {steps*dt} seconds")
start_time = time.time()
for _ in range(n_simulations):
    humans_state = jnp.copy(initial_humans_state)
    for _ in range(steps):
        filtered_static_obstacles = filter_obstacles(humans_state, new_static_obstacles, static_obstacles_per_cell, grid_coords, grid_cell_size)
        humans_state = step(humans_state, humans_goal, humans_parameters, filtered_static_obstacles, dt)
        humans_state.block_until_ready() # Wait for the computation to finish
end_time = time.time()
print("Simulations done! Average execution time per simulation: ", (end_time - start_time)/n_simulations)

# Simulation saving state for plotting
humans_state = jnp.copy(initial_humans_state)
all_states = np.empty((steps+1, len(humans_state), 6), np.float32)
all_states[0] = humans_state
for i in range(steps):
    filtered_static_obstacles = filter_obstacles(humans_state, new_static_obstacles, static_obstacles_per_cell, grid_coords, grid_cell_size)
    humans_state = step(humans_state, humans_goal, humans_parameters, filtered_static_obstacles, dt)
    all_states[i+1] = humans_state
end_time = time.time()
all_states = jax.device_get(all_states) # Transfer data from GPU to CPU for plotting (only at the end)

# Plot
COLORS = list(mcolors.TABLEAU_COLORS.values())
print("\nPlotting...")
figure, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')
ax.set(xlabel='X',ylabel='Y')
# Plot the grid given computed grid_coords
for coord in grid_coords.reshape(-1,2):
    rect = plt.Rectangle((coord[0], coord[1]), grid_cell_size, grid_cell_size, facecolor='none', edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=0)
    ax.add_patch(rect)
for h in range(len(humans_state)): 
    ax.plot(all_states[:,h,0], all_states[:,h,1], color=COLORS[h%len(COLORS)], linewidth=0.5, zorder=0)
    ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=COLORS[h%len(COLORS)], zorder=2)
    for k in range(0,steps+1,int(3/dt)):
        head = plt.Circle((all_states[k,h,0] + np.cos(all_states[k,h,4]) * humans_parameters[h,0], all_states[k,h,1] + np.sin(all_states[k,h,4]) * humans_parameters[h,0]), 0.1, color=COLORS[h%len(COLORS)], zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((all_states[k,h,0],all_states[k,h,1]),humans_parameters[h,0], edgecolor=COLORS[h%len(COLORS)], facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
        num = int(k*dt) if (k*dt).is_integer() else (k*dt)
        ax.text(all_states[k,h,0],all_states[k,h,1], f"{num}", color=COLORS[h%len(COLORS)], va="center", ha="center", size=10, zorder=1, weight='bold')
for o in static_obstacles: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
if not os.path.exists(os.path.join(os.path.dirname(__file__),".images")):
    os.makedirs(os.path.join(os.path.dirname(__file__),".images"))
figure.savefig(os.path.join(os.path.dirname(__file__),".images",f"example7.png"), format='png')
plt.show()