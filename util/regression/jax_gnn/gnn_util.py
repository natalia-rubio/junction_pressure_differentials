import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np
np.random.seed(0)
import pdb

# Neural Network Architecture
def get_sizes(network_params):
    # Get the sizes of the network layers

    encoder_sizes = [network_params["num_input_features"]] + [network_params["layer_width"]]*network_params["num_encoder_layers"]
    passing_sizes = [2*network_params["layer_width"]] + [network_params["layer_width"]]*network_params["num_passing_layers"]
    decoder_sizes = [network_params["layer_width"]]*network_params["num_decoder_layers"] + [network_params["num_output_features"]]
    return encoder_sizes, passing_sizes, decoder_sizes

def random_layer_params(m, n, key, scale=1e-2):
    # Randomly initialize the weights of a layer
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_weights(network_params):
    # Initialize the weights of the network
    key = random.key(0)
    encoder_sizes, passing_sizes, decoder_sizes = get_sizes(network_params)

    encoder_weights = [random_layer_params(m, n, k) for m, n, k in zip(encoder_sizes[:-1], encoder_sizes[1:], random.split(key, len(encoder_sizes)))]
    passing_weights = [random_layer_params(m, n, k) for m, n, k in zip(passing_sizes[:-1], passing_sizes[1:], random.split(key, len(passing_sizes)))]
    decoder_weights = [random_layer_params(m, n, k) for m, n, k in zip(decoder_sizes[:-1], decoder_sizes[1:], random.split(key, len(decoder_sizes)))]
    return encoder_weights, passing_weights, decoder_weights


# Data Handling
def get_train_val_split(num_pts, percent_train=0.8):
    # Split the data into training and validation sets
    indices = np.random.permutation(num_pts)
    train_indices = indices[:int(percent_train*num_pts)]
    val_indices = indices[int(percent_train*num_pts):]
    return train_indices, val_indices

def get_batch_indices(indices, batch_size):
    # Split data set into random batches
    num_batches = len(indices)//batch_size
    np.random.shuffle(indices)
    indices = indices[:num_batches*batch_size] # drop the last few if they don't make a full batch
    return [indices[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

# Network Utilities
def inv_scale_jax(scaling_dict, field, field_name):
    # Inverse data normalization function
    mean = scaling_dict[field_name][0]
    std = scaling_dict[field_name][1]
    scaled_field = jnp.add(jnp.multiply(field, std), mean)
    return jnp.reshape(scaled_field, (-1,1))

def relu(x):
  # Rectified Linear Unit activation function
  return jnp.maximum(0, x)

    
def forward_pass(input,weights):
    # Forward pass through the network
    latent_rep = input    
    for w, b in weights[:-1]:
        lin_comb = jnp.dot(w, latent_rep) + b
        latent_rep = relu(lin_comb)

    final_w, final_b = weights[-1]
    output = jnp.dot(final_w, latent_rep) + final_b
    return output

batched_forward_pass = vmap(forward_pass, in_axes=(0, None))

# Optimization
# opt_state = optimizer.init(params)