from jax import numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from jhsfm.hsfm import full_update as update

COLORS = list(mcolors.TABLEAU_COLORS.values())

# Hyperparameters
room_half_length = 7
dt = 0.01
steps = 1500
humans_state = np.array([[7.,0.,0.,0.,jnp.pi,0.],
                         [6.8,0.8,0.,0.,jnp.pi,0.],
                         [6.8,-0.8,0.,0.,jnp.pi,0.],
                         [6.5,1.5,0.,0.,jnp.pi,0.],
                         [6.5,-1.5,0.,0.,jnp.pi,0.]])
static_obstacles = jnp.array([[[[-0.1,0.5],[0.1,0.5]],[[0.1,0.5],[0.1,3]],[[0.1,3],[-0.1,3]],[[-0.1,3],[-0.1,0.5]]],
                              [[[-0.1,-0.5],[0.1,-0.5]],[[0.1,-0.5],[0.1,-3]],[[0.1,-3],[-0.1,-3]],[[-0.1,-3],[-0.1,-0.5]]]])


# Initial conditions
humans_parameters = np.zeros((len(humans_state), 19))
humans_goal = np.zeros((len(humans_state), 2))
for i in range(len(humans_state)):
    # Goal: (gx, gy)
    humans_goal[i,0] = -room_half_length
    humans_goal[i,1] = 0.
    # Parameters: (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
    humans_parameters[i,0] = 0.3
    humans_parameters[i,1] = 80.
    humans_parameters[i,2] = 1.
    humans_parameters[i,3] = 0.5
    humans_parameters[i,4] = 2000.
    humans_parameters[i,5] = 2000.
    humans_parameters[i,6] = 0.08
    humans_parameters[i,7] = 0.08
    humans_parameters[i,8] = 120.
    humans_parameters[i,9] = 120.
    humans_parameters[i,10] = 0.6
    humans_parameters[i,11] = 0.6
    humans_parameters[i,12] = 120000.
    humans_parameters[i,13] = 240000.
    humans_parameters[i,14] = 1.
    humans_parameters[i,15] = 500.
    humans_parameters[i,16] = 3.
    humans_parameters[i,17] = 0.1
    humans_parameters[i,18] = 0.
humans_state = jnp.array(humans_state)
humans_parameters = jnp.array(humans_parameters)
humans_goal = jnp.array(humans_goal)

# Simulation
print(f"\nAvailable devices: {jax.devices()}\n")
print("Starting simulation...\n")
start_time = time.time()
all_states = np.empty((steps+1, len(humans_state), 6), np.float32)
all_states[0] = humans_state
for i in range(steps):
    humans_state = update(humans_state, humans_goal, humans_parameters, static_obstacles, dt)
    all_states[i+1] = humans_state
end_time = time.time()
print("Simulation done! Total time: ", end_time - start_time)
all_states = jax.device_get(all_states) # Transfer data from GPU to CPU for plotting (only at the end)

# Plot
print("\nPlotting...")
figure, ax = plt.subplots()
ax.axis('equal')
ax.set(xlabel='X',ylabel='Y',xlim=[-room_half_length-1,room_half_length+1],ylim=[-room_half_length-1,room_half_length+1])
for h in range(len(humans_state)): 
    ax.plot(all_states[:,h,0], all_states[:,h,1], color=COLORS[h%len(COLORS)], linewidth=0.5, zorder=0)
    ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=COLORS[h%len(COLORS)], zorder=2)
    for k in range(0,steps+1,300):
        head = plt.Circle((all_states[k,h,0] + np.cos(all_states[k,h,4]) * humans_parameters[h,0], all_states[k,h,1] + np.sin(all_states[k,h,4]) * humans_parameters[h,0]), 0.1, color=COLORS[h%len(COLORS)], zorder=1)
        ax.add_patch(head)
        circle = plt.Circle((all_states[k,h,0],all_states[k,h,1]),humans_parameters[h,0], edgecolor=COLORS[h%len(COLORS)], facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
        num = int(k*dt) if (k*dt).is_integer() else (k*dt)
        ax.text(all_states[k,h,0],all_states[k,h,1], f"{num}", color=COLORS[h%len(COLORS)], va="center", ha="center", size=10, zorder=1, weight='bold')
for o in static_obstacles: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
plt.show()