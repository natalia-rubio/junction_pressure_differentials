import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from util.tools.basic import *
from util.regression.jax_nn.nn_util import *
import time

import time

def train_nn(model, anatomy, set_type, training_params):
    
    train_ind, val_ind = get_train_val_split(len(model.data_dict["geo"])) # Split the data into training and validation sets

    for epoch in range(training_params['num_epochs']): # Loop through the epochs
        start_time = time.time() # Time each epoch
        batch_ind_list = get_batch_indices(train_ind, training_params['batch_size']) # Split the training set into random batches
        for i, batch_inds in enumerate(batch_ind_list): # Loop through the batches
            model.update(batch_inds, model.weights, training_params["step_size"]) # Update the model based on the batch
        epoch_time = time.time() - start_time

    # Print epoch results
    train_loss = model.loss(train_ind, model.weights)
    test_loss  = model.loss(val_ind, model.weights)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_loss))
    print("Test set accuracy {}".format(test_loss))

    return train_loss, test_loss