import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
from util.tools.basic import *
from util.regression.jax_nn.nn_util import *
from util.regression.jax_nn.nn_model import *
import time

import time

def train_nn(model, anatomy, set_type, training_params):
    
    train_inds, val_inds = get_train_val_split(len(model.data_dict["geo"])) # Split the data into training and validation sets
    #optimizer = optax.adam(learning_rate = training_params["step_size"])

    for epoch in range(training_params['num_epochs']): # Loop through the epochs
        start_time = time.time() # Time each epoch
        batch_ind_list = get_batch_indices(train_inds, training_params['batch_size']) # Split the training set into random batches
        for i, batch_inds in enumerate(batch_ind_list): # Loop through the batches
            model.weights = update_model(input = model.data_dict["geo"][batch_inds,:],
                         flow =  jnp.reshape(model.data_dict["flow"][batch_inds,0:4], (-1,4)),
                         dP_true = jnp.reshape(model.data_dict["dP"][batch_inds,0:4], (-1,4)),
                         scaling_dict = model.scaling_dict,
                         weights = model.weights, 
                         step_size = training_params["step_size"]) # Update the model based on the batch
        epoch_time = time.time() - start_time

        # Print epoch results
        train_loss = loss(input = model.data_dict["geo"][train_inds,:],
                        flow =  jnp.reshape(model.data_dict["flow"][train_inds,0:4], (-1,4)),
                        dP_true = jnp.reshape(model.data_dict["dP"][train_inds,0:4], (-1,4)),
                        scaling_dict = model.scaling_dict,
                        weights = model.weights)
        val_loss = loss(input = model.data_dict["geo"][val_inds,:],
                        flow =  jnp.reshape(model.data_dict["flow"][val_inds,0:4], (-1,4)),
                        dP_true = jnp.reshape(model.data_dict["dP"][val_inds,0:4], (-1,4)),
                        scaling_dict = model.scaling_dict,
                        weights = model.weights)
        print("Epoch {} in {:0.2f} sec  |  ".format(epoch, epoch_time) + \
              "Training set accuracy {:e}  |  ".format(train_loss) + \
              "Validation set accuracy {:e}".format(val_loss))

    return train_loss, val_loss