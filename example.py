from jax import numpy as jnp
import numpy as np
from jhsfm.hsfm import full_update as update

# Hyperparameters
n_humans = 5
circle_radius = 5
dt = 0.01
steps = 100

# Initialize conditions
humans_state = np.zeros((n_humans, 6))
humans_parameters = np.zeros((n_humans, 19))
humans_goal = np.zeros((n_humans, 2))
angle_width = (2 * jnp.pi) / (n_humans)
for i in range(n_humans):
    # State: (px, py, bvx, bvy, theta, omega)
    humans_state[i,0] = circle_radius * jnp.cos(i * angle_width)
    humans_state[i,1] = circle_radius * jnp.sin(i * angle_width)
    humans_state[i,2] = 0
    humans_state[i,3] = 0
    humans_state[i,4] = -jnp.pi + i * angle_width
    humans_state[i,5] = 0
    # Goal: (gx, gy)
    humans_goal[i,0] = -humans_state[i,0]
    humans_goal[i,1] = -humans_state[i,1]
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
all_states = []
for i in range(steps):
    humans_state = update(humans_state, humans_goal, humans_parameters, dt)
    all_states.append(humans_state)