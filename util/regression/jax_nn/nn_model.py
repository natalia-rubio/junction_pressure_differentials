import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from util.tools.basic import *
from util.regression.jax_nn.nn_util import *

class NeuralNet():
   
    def __init__(self, network_params):
        self.anatomy = network_params["anatomy"]; self.set_type = network_params["set_type"]
        self.data_dict = load_dict(f"data/numpy_arrays/{self.anatomy}/{self.set_type}/{self.anatomy}_{self.set_type}_data_dict") # Load all data
        self.scaling_dict = load_dict(f"data/scaling_dictionaries/{self.anatomy}_{self.set_type}_scaling_dict")
        self.weights = init_weights(network_params)
        return

    @jit
    def predict(indices, weights):
    # per-example predictions
        input = self.data_dict["geo"][indices,:]
        output = batched_forward_pass(input, weights)
        return output
    

    def loss(self, indices, weights):
        flow = jnp.reshape(self.data_dict["flow"][indices,0:4], (-1,4))
        dP_true = jnp.reshape(self.data_dict["dP"][indices,0:4], (-1,4))
        
        coefs_pred = self.predict(indices, weights)
        dP_pred = inv_scale_jax(self.scaling_dict, coefs_pred[:,0], "coef_a"), jnp.square(flow) + \
            inv_scale_jax(self.scaling_dict, coefs_pred[:,1], "coef_b") * flow
        pdb.set_trace()
        return jnp.sqrt(jnp.mean(jnp.square(dP_pred - dP_true)))

    # @jit
    def update(self, indices, weights, step_size):
        grads = grad(self.loss, argnums = 1)(indices, weights)
        self.weights = [(w - step_size * dw, b - step_size * db)
                        for (w, b), (dw, db) in zip(weights, grads)]