import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
from util.tools.basic import *
from util.regression.jax_gnn.gnn_util import *
from util.regression.jax_gnn.gnn_model import *
import time

import time

def train_gnn(model, anatomy, set_type, training_params):
    
    train_inds, val_inds = get_train_val_split(len(model.graph_list)) # Split the data into training and validation sets
    
    for epoch in range(training_params['num_epochs']): # Loop through the epochs
        start_time = time.time() # Time each epoch
        batch_ind_list = get_batch_indices(train_inds, training_params['batch_size']) # Split the training set into random batches

        for i, batch_inds in enumerate(batch_ind_list): # Loop through the batches
            model.update(indices = batch_inds) # Update the model based on the batch
        
        epoch_time = time.time() - start_time

        # Print epoch results
        
        train_loss = loss(graph = jraph.batch([model.graph_list[i] for i in list(train_inds)]),
                        scaling_dict = model.scaling_dict,
                        encoder_weights = model.encoder_weights,
                        update_weights = model.passing_weights, 
                        main_path_decoder_weights = model.main_path_decoder_weights, 
                        aux_path_decoder_weights= model.aux_path_decoder_weights,
                        num_message_passing_steps = model.num_message_passing_steps)
        val_loss = loss(graph = jraph.batch([model.graph_list[i] for i in list(val_inds)]),
                        scaling_dict = model.scaling_dict,
                        encoder_weights = model.encoder_weights,
                        update_weights = model.passing_weights, 
                        main_path_decoder_weights = model.main_path_decoder_weights, 
                        aux_path_decoder_weights= model.aux_path_decoder_weights,
                        num_message_passing_steps = model.num_message_passing_steps)
        print("Epoch {} in {:0.2f} sec  |  ".format(epoch, epoch_time) + \
              "Training set accuracy {:e}  |  ".format(train_loss) + \
              "Validation set accuracy {:e}".format(val_loss))

    return train_loss, val_loss