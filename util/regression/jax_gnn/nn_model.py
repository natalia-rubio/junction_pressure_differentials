import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from util.tools.basic import *
from util.regression.jax_nn.nn_util import *
import optax
import haiku as hk
import jraph

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
            jnp.reshape(self.data_dict["flow"][indices,0:4], (-1,4)),
            jnp.reshape(self.data_dict["dP"][indices,0:4], (-1,4)),
            self.scaling_dict,
            self.weights)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.weights = optax.apply_updates(self.weights, updates)
        return


def hookes_hamiltonian_from_graph_fn(
    graph: jraph.GraphsTuple) -> jraph.GraphsTuple:

#   def update_edge_fn(edges, sent_attributes, received_attributes, global_edge_attributes):
#     input = jnp.concatenate([edges, sent_attributes, received_attributessenders, globals_])
#     layer_sizes = network_params["num_edge_features"] + network_params["num_hidden_layers"] * [network_params["latent_size"]]
#     return hk.nets.MLP(layer_sizes)(input)

    def update_node_fn(nodes, sent_attributes, received_attributes, global_attributes):
        input = jnp.concatenate([nodes, sent_attributes, received_attributes, global_attributes])
        layer_sizes = network_params["num_edge_features"] + network_params["num_hidden_layers"] * [network_params["latent_size"]]
        return hk.nets.MLP(layer_sizes)(input)
    
    def aggregate_edges_for_nodes_fn(e, edge_gr_idx, sum_n_node):
        return jraph.segment_sum
    def aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph):
        return jraph.segment_sum
    def aggregate_edges_for_globals_fn(n, node_gr_idx, n_graph):
        return jraph.segment_sum
    
    def update_global_fn(nodes, edges, globals_):
        del globals_
        # At this point we will receive node and edge features aggregated (summed)
        # for all nodes and edges in each graph.
        hamiltonian_per_graph = nodes["kinetic_energy"] + edges["hookes_potential"]
        return frozendict({"hamiltonian": hamiltonian_per_graph})

    gn = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn)

    return gn(graph)

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
