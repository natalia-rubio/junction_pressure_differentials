# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library of Graph Neural Network models."""

import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Union
import pdb
import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jraph._src import graph as gn_graph
from jraph._src import utils
import jraph

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from util.tools.basic import *
from util.regression.jax_gnn.gnn_util import *
import optax

class GraphNeuralNet():
   
    def __init__(self, network_params, optimizer_params):
        self.anatomy = network_params["anatomy"]; self.set_type = network_params["set_type"]
        # self.data_dict = load_dict(f"data/numpy_arrays/{self.anatomy}/{self.set_type}/{self.anatomy}_{self.set_type}_data_dict") # Load all data
        # self.graph = load_dict(f"data/synthetic_junctions_reduced_results/AP/random/AP_000/graph")
        self.graph_list = load_dict(f"data/graphs/{self.anatomy}/{self.set_type}/graph_list")
        self.scaling_dict = load_dict(f"data/scaling_dictionaries/{self.anatomy}_{self.set_type}_scaling_dict")
        self.encoder_weights, self.passing_weights, self.decoder_weights = init_weights(network_params)
        self.num_message_passing_steps = network_params["num_message_passing_steps"]

        self.scheduler = optax.exponential_decay(init_value = optimizer_params["init"], 
                                                 transition_steps = optimizer_params["transition_steps"], 
                                                 decay_rate = optimizer_params["decay_rate"])
        self.optimizer = optax.adam(learning_rate = self.scheduler)
        self.opt_state = self.optimizer.init((self.encoder_weights, self.passing_weights, self.decoder_weights))
        return
    
    def update(self, indices):
        graph = jraph.batch([self.graph_list[i] for i in list(indices)])
        grads = grad(loss, argnums = [-4, -3, -2])(graph, #(self.data_dict["geo"][indices,:],
            self.scaling_dict,
            self.encoder_weights, self.passing_weights, self.decoder_weights, self.num_message_passing_steps)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.encoder_weights, self.passing_weights, self.decoder_weights = optax.apply_updates(
            (self.encoder_weights, self.passing_weights, self.decoder_weights), updates)
        return


#@jit
def loss(graph, scaling_dict, encoder_weights, update_weights, decoder_weights, num_message_passing_steps):

    flow = graph.globals[:,3:6]; dP_true = graph.globals[:,6:9]
    coefs_pred = predict(graph, encoder_weights, update_weights, decoder_weights, num_message_passing_steps)
    # print(coefs_pred)
    # pdb.set_trace()

    dP_pred = inv_scale_jax(scaling_dict, coefs_pred[:,0], "coef_a") * jnp.square(flow) + inv_scale_jax(scaling_dict, coefs_pred[:,1], "coef_b") * flow
    return jnp.sqrt(jnp.mean(jnp.square((dP_pred - dP_true)/1333)))

#@jit
def predict(graph, encoder_weights, passing_weights, decoder_weights, num_message_passing_steps):
    """
    Args:
      graph: a `GraphsTuple` containing the graph.

    Returns:
      aggregated characteristic values
    """
    # pylint: disable=g-long-lambda
    nodes, edges, receivers, senders, globals, n_node, n_edge = graph
    node_types = nodes[:,3:6]
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    sum_n_edge = senders.shape[0]
    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
      raise ValueError(
          'All node arrays in nest must contain the same number of nodes.')
    nodes = batched_relu(batched_forward_pass(nodes, encoder_weights))

    for passing_step in range(num_message_passing_steps):
        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
        
        updated_nodes = batched_relu(batched_forward_pass(jnp.concatenate([nodes[receivers], received_attributes],axis = 1), passing_weights))
        nodes.at[receivers].set(updated_nodes)

    nodes = batched_forward_pass(nodes, decoder_weights)
    #pdb.set_trace()
    nodes = jnp.multiply(nodes, (node_types[:,-1] == 0).astype(jnp.float32).reshape(-1,1))
    #print(nodes)
    return jnp.sum(nodes.reshape(-1, 29, 3), axis=1)