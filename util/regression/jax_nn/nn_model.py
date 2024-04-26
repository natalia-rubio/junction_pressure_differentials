import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from util.tools.basic import *
from util.regression.jax_nn.nn_util import *
import optax

class NeuralNet():
   
    def __init__(self, network_params):
        self.anatomy = network_params["anatomy"]; self.set_type = network_params["set_type"]
        self.data_dict = load_dict(f"data/numpy_arrays/{self.anatomy}/{self.set_type}/{self.anatomy}_{self.set_type}_data_dict") # Load all data
        self.scaling_dict = load_dict(f"data/scaling_dictionaries/{self.anatomy}_{self.set_type}_scaling_dict")
        self.weights = init_weights(network_params)
        return

#input = self.data_dict["geo"][indices,:]
@jit
def predict(input, weights):
    output = batched_forward_pass(input, weights)
    return output

@jit
def loss(input, flow, dP_true, scaling_dict, weights):
    coefs_pred = predict(input, weights)
    dP_pred = inv_scale_jax(scaling_dict, coefs_pred[:,0], "coef_a") * jnp.square(flow) + \
        inv_scale_jax(scaling_dict, coefs_pred[:,1], "coef_b") * flow
    return jnp.sqrt(jnp.mean(jnp.square((dP_pred - dP_true)/1333)))


def update_model(input, flow, dP_true, scaling_dict, weights, optimizer, opt_state):
    grads = grad(loss, argnums = -1)(input, 
                                    flow, 
                                    dP_true,
                                    scaling_dict,
                                    weights)
    # new_weights = [(w - step_size * dw, b - step_size * db)
    #                 for (w, b), (dw, db) in zip(weights, grads)]
    updates, opt_state = optimizer.update(grads, opt_state)
    new_weights = optax.apply_updates(weights, updates)
    return new_weights, opt_state