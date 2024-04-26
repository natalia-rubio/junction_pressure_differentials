import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np
np.random.seed(0)

# Neural Network Architecture
def get_sizes(network_params):
    num_input_features = network_params["num_input_features"]
    num_layers = network_params["num_layers"]
    layer_width = network_params["layer_width"]
    num_output_features = network_params["num_output_features"]
    sizes = [num_input_features] + [layer_width]*num_layers + [num_output_features]
    return sizes

def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_weights(network_params):
    key = random.key(0)
    sizes = get_sizes(network_params)
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


# Data Handling
def get_train_val_split(num_pts, percent_train=0.8):
    indices = np.random.permutation(num_pts)
    train_indices = indices[:int(percent_train*num_pts)]
    val_indices = indices[int(percent_train*num_pts):]
    return train_indices, val_indices

def get_batch_indices(indices, batch_size):
    num_batches = len(indices)//batch_size
    np.random.shuffle(indices)
    indices = indices[:num_batches*batch_size] # drop the last few if they don't make a full batch
    return [indices[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

# Prediction and Loss Functions
def inv_scale_jax(scaling_dict, field, field_name):
    mean = scaling_dict[field_name][0]
    std = scaling_dict[field_name][1]
    scaled_field = jnp.add(jnp.multiply(field, std), mean)
    return jnp.reshape(scaled_field, (-1,1))

def relu(x):
  return jnp.maximum(0, x)

def forward_pass(input,weights):
    latent_rep = input    
    for w, b in weights[:-1]:
        lin_comb = jnp.dot(w, latent_rep) + b
        latent_rep = relu(lin_comb)

    final_w, final_b = weights[-1]
    output = jnp.dot(final_w, latent_rep) + final_b
    return output

batched_forward_pass = vmap(forward_pass, in_axes=(0, None))