import jax.numpy as jnp
from jax import jit, vmap, lax, debug, random

# TODO: Add generate random humans parameters function
# TODO: Add generate random circular crossing scenario initial conditions function
# TODO: Add generate random parallel traffica initial conditions function

def get_standard_humans_parameters(n_humans:int):
    """
    Returns the standard parameters of the HSFM for the humans in the simulation. Parameters are the same for all humans in the form:
    (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)

    args:
    - n_humans: int - Number of humans in the simulation.

    outputs:
    - parameters (n_humans, 19) - Standard parameters for the humans in the simulation.
    """
    single_params = jnp.array([0.3, 80., 1., 0.5, 2000., 2000., 0.08, 0.08, 120., 120., 0.6, 0.6, 120000., 240000., 1., 500., 3., 0.1, 0.])
    return jnp.tile(single_params, (n_humans, 1))