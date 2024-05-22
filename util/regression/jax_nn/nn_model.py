import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from util.tools.basic import *
from util.regression.jax_nn.nn_util import *
import optax

class NeuralNet():
   
    def __init__(self, network_params, optimizer_params):
        self.anatomy = network_params["anatomy"]; self.set_type = network_params["set_type"]
        self.data_dict = load_dict(f"data/numpy_arrays/{self.anatomy}/{self.set_type}/{self.anatomy}_{self.set_type}_data_dict") # Load all data
        self.scaling_dict = load_dict(f"data/scaling_dictionaries/{self.anatomy}_{self.set_type}_scaling_dict")
        self.weights = init_weights(network_params)

        # rate_factor = ((count - transition_begin) / transition_steps)
        # decayed_value = init_value * (decay_rate ** rate_factor)
        self.scheduler = optax.exponential_decay(init_value = optimizer_params["init"], 
                                                 transition_steps = optimizer_params["transition_steps"], 
                                                 decay_rate = optimizer_params["decay_rate"])
        self.optimizer = optax.adam(learning_rate = self.scheduler)
        self.opt_state = self.optimizer.init(self.weights)
        return
    
    def update(self, indices):
        grads = grad(loss, argnums = -1)(self.data_dict["geo"][indices,:],
            self.data_dict["flow"][indices,:],
            self.data_dict["dP"][indices,:],
            self.scaling_dict,
            self.weights)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.weights = optax.apply_updates(self.weights, updates)
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
