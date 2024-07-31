import jax.numpy as jnp
from jax import jit, vmap, lax, debug
import numpy as np

@jit
def wrap_angle(theta:jnp.float32) -> jnp.float32:
    """
    This function wraps the angle to the interval [-pi, pi]
    
    args:
    - theta: angle to be wrapped
    
    output:
    - wrapped_theta: angle wrapped to the interval [-pi, pi]
    """
    wrapped_theta = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return wrapped_theta

@jit
def get_linear_velocity(theta:jnp.float32, body_velocity: jnp.ndarray) -> jnp.ndarray:
    """
    This function computes the linear velocity of the agent in the world frame
    
    args:
    - theta: angle of the agent in the world frame
    - body_velocity: velocity of the agent in its body frame
    
    output: 
    - linear_velocity: velocity of the agent in the world frame
    """
    rotational_matrix = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    linear_velocity = jnp.matmul(rotational_matrix, body_velocity)
    return linear_velocity

@jit
def compute_closest_point_to_the_edge(obs_idx:jnp.int32, reference_point:jnp.ndarray, edge:jnp.ndarray, current_closest_point:jnp.ndarray, current_min_distance:jnp.ndarray) -> jnp.ndarray:
    """
    This function computes the closest point of the edge to the reference point and confronts it with the current closest point and min dist to the obstacle.
    Finally it overweites the closest point and the min distance to the obstacle in the current ones.
    
    args:
    - obs_idx: index of the obstacle
    - reference_point: shape is (2,) in the form (px, py)
    - edge: shape is (2, 2) where each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - current_closest_point: shape is (2,) in the form (cx, cy)
    - current_min_distance: min distances to the current closest point

    output:
    - None
    """
    a = edge[0]
    b = edge[1]
    t = (jnp.dot(reference_point - a, b - a)) / (jnp.linalg.norm(b - a) ** 2)
    t_lb = lax.cond(t>0,lambda x: x,lambda x: 0.,t)
    t_star = lax.cond(t_lb<1,lambda x: x,lambda x: 1.,t_lb)
    h = a + t_star * (b - a)
    dist = jnp.linalg.norm(h - reference_point)
    current_closest_point = current_closest_point.at[obs_idx,:].set((dist < current_min_distance[obs_idx], lambda x: h, lambda x: x, current_closest_point[obs_idx]))
    current_min_distance = current_min_distance.at[obs_idx].set(lax.cond(dist < current_min_distance[obs_idx], lambda x: dist, lambda x: x, current_min_distance[obs_idx]))

@jit
def full_update(humans_state:jnp.ndarray, humans_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:jnp.float32) -> jnp.ndarray:
    """
    This functions makes a step in time (of length dt) for the humans' state using the Headed Social Force Model (HSFM) with 
    global force guidance for torque and sliding component on the repulsive forces.

    args:
    - humans_state: shape is (n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
    - humans_goal: shape is (n_humans, 2) where each row is (gx, gy)
    - parameters: shape is (n_humans, 19) where each row is (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
    - obstacles: shape is (n_obstacles, n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - dt: sampling time for the update
    
    output:
    - updated_humans_state: shape is (n_humans, 6) where each row is (px, py, bvx, bvy, theta, omega)
    """
    stacked_states = jnp.stack([humans_state for _ in range(len(humans_state))], axis=0)
    stacked_parameters = jnp.stack([parameters for _ in range(len(humans_state))], axis=0)
    stacked_obstacles = jnp.stack([obstacles for _ in range(len(humans_state))], axis=0)
    idxs = jnp.arange(len(humans_state))
    dts = jnp.ones((len(humans_state),)) * dt
    updated_humans_state = single_update(idxs, stacked_states, humans_goal, stacked_parameters, stacked_obstacles, dts)
    return updated_humans_state

@vmap
def single_update(idx:jnp.int32, humans_state:jnp.ndarray, human_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:jnp.float32) -> jnp.ndarray:
    """
    This functions makes a step in time (of length dt) for a single human using the Headed Social Force Model (HSFM) with 
    global force guidance for torque and sliding component on the repulsive forces.

    args:
    - idx: human index in the state, goal and parameter vectors
    - humans_state: shape is (n_humans, 6) in the form is (px, py, bvx, bvy, theta, omega)
    - humans_goal: shape is (2,) in the form (gx, gy)
    - parameters: shape is (n_humans, 19) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
    - obstacles: shape is (n_obstacles, n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - dt: sampling time for the update
    
    output:
    - updated_human_state: shape is (6,) in the form (px, py, bvx, bvy, theta, omega)
    """
    self_state = humans_state[idx]
    self_parameters = parameters[idx]
    # Desired force computation
    linear_velocity = get_linear_velocity(self_state[4], self_state[2:4])
    diff = human_goal - self_state[:2]
    dist = jnp.linalg.norm(diff)
    desired_force =  lax.cond(dist > self_parameters[0],lambda x: x * (self_parameters[1] * (((diff / dist) * self_parameters[2]) - linear_velocity) / self_parameters[3]),lambda x: x * 0,jnp.ones((2,)))
    # Social force computation
    social_force = lax.fori_loop(0, len(humans_state), lambda j, acc: lax.cond(j != idx, lambda acc: acc + pairwise_social_force(self_state, humans_state[j], self_parameters, parameters[j]), lambda acc: acc, acc), jnp.zeros((2,)))
    # Obstacle force computation
    closest_points = jnp.zeros((len(obstacles), 2))
    min_distances = jnp.ones((len(obstacles),)) * 10000
    for i, o in enumerate(obstacles):
        for edge in o:
            a = edge[0]
            b = edge[1]
            t = (jnp.dot(self_state[0:2] - a, b -a)) / (jnp.linalg.norm(b - a) ** 2)
            t_lb = lax.cond(t>0,lambda x: x,lambda x: 0.,t)
            t_star = lax.cond(t_lb<1,lambda x: x,lambda x: 1.,t_lb)
            h = a + t_star * (b - a)
            dist = jnp.linalg.norm(h - self_state[0:2])
            closest_points = closest_points.at[i,:].set(lax.cond(dist < min_distances[i], lambda x: h, lambda x: x, closest_points[i]))
            min_distances = min_distances.at[i].set(lax.cond(dist < min_distances[i], lambda x: dist, lambda x: x, min_distances[i]))
    obstacle_force = jnp.zeros((2,))
    # Torque computation
    input_force = desired_force + social_force + obstacle_force
    input_force_norm = jnp.linalg.norm(input_force)
    input_force_angle = jnp.arctan2(input_force[1], input_force[0])
    inertia = (self_parameters[1] * self_parameters[0] * self_parameters[0]) / 2
    k_theta = inertia * self_parameters[17] * input_force_norm
    k_omega = inertia * (1 + self_parameters[16]) * jnp.sqrt((self_parameters[17] * input_force_norm) / self_parameters[16])
    torque = - k_theta * wrap_angle(self_state[4] - input_force_angle) - k_omega * self_state[5]
    # Global force computation
    global_force = jnp.zeros((2,))
    global_force = global_force.at[0].set(jnp.dot(input_force, jnp.array([jnp.cos(self_state[4]), jnp.sin(self_state[4])])))
    global_force = global_force.at[1].set(self_parameters[14] * jnp.dot(social_force + obstacle_force, jnp.array([-jnp.sin(self_state[4]), jnp.cos(self_state[4])])) - self_parameters[15] * self_state[3])
    # Update
    updated_human_state = jnp.zeros((6,))
    updated_human_state = updated_human_state.at[0].set(self_state[0] + dt * linear_velocity[0])
    updated_human_state = updated_human_state.at[1].set(self_state[1] + dt * linear_velocity[1])
    updated_human_state = updated_human_state.at[4].set(wrap_angle(self_state[4] + dt * self_state[5]))
    updated_human_state = updated_human_state.at[2].set(self_state[2] + dt * (global_force[0] / self_parameters[1]))
    updated_human_state = updated_human_state.at[3].set(self_state[3] + dt * (global_force[1] / self_parameters[1]))
    updated_human_state = updated_human_state.at[5].set(self_state[5] + dt * (torque / inertia))
    # DEBUGGING
    # debug.print("\n")
    # debug.print("jax.debug.print(closest_points) -> {x}", x=closest_points)
    # debug.print("jax.debug.print(min_distances) -> {x}", x=min_distances)
    # debug.print("jax.debug.print(torque) -> {x}", x=torque)
    # debug.print("jax.debug.print(input_force) -> {x}", x=input_force)
    # debug.print("jax.debug.print(desired_force) -> {x}", x=desired_force)
    # debug.print("jax.debug.print(social_force) -> {x}", x=social_force)
    # debug.print("jax.debug.print(obstacle_force) -> {x}", x=obstacle_force)
    # debug.print("jax.debug.print(global_force) -> {x}", x=global_force)
    # debug.print("jax.debug.print(updated_human_state) -> {x}", x=updated_human_state)
    return updated_human_state

@jit
def pairwise_social_force(human_state:jnp.ndarray, other_human_state:jnp.ndarray, parameters:jnp.ndarray, other_human_parameters:jnp.ndarray):
    """
    This function computes the social force between a pair of humans

    args:
    - human_state: shape is (6,) in the form (px, py, bvx, bvy, theta, omega)
    - other_humans_state: shape is (6,) in the form (px, py, bvx, bvy, theta, omega)
    - parameters: shape is (19,) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, k0, kd, alpha, k_lambda, safety_space)
    - other_humans_parameters: shape is (19,) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, k0, kd, alpha, k_lambda, safety_space)

    output:
    - social_force: shape is (2,) in the form (fx, fy)
    """
    rij = parameters[0] + other_human_parameters[0] + parameters[18] + other_human_parameters[18]
    diff = human_state[:2] - other_human_state[:2]
    dist = jnp.linalg.norm(diff)
    nij = diff / dist
    real_dist = rij - dist
    tij = jnp.array([-nij[1], nij[0]])
    human_linear_velocity = get_linear_velocity(human_state[4], human_state[2:4])
    other_human_linear_velocity = get_linear_velocity(other_human_state[4], other_human_state[2:4])
    delta_vij = jnp.dot(other_human_linear_velocity - human_linear_velocity, tij)
    pairwise_social_force = lax.cond(real_dist > 0, lambda x: x * (parameters[4] * jnp.exp(real_dist / parameters[6]) + parameters[12] * real_dist) * nij + (parameters[8] * jnp.exp(real_dist / parameters[10]) + parameters[13] * real_dist * delta_vij) * tij, lambda x: x * (parameters[4] * jnp.exp(real_dist / parameters[6])) * nij + (parameters[8] * jnp.exp(real_dist / parameters[10])) * tij, jnp.ones((2,)))
    return pairwise_social_force


