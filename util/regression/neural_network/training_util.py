import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
import dgl
print(dgl.__version__)
#from util.regression.neural_network.graphnet_nn import GraphNet
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from datetime import datetime
import random
import time
import json
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import random
random.seed(10)
font = {"size"   : 14}

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(dir):
    f = open(dir, "rb"); scaling_dict = pickle.load(f)
    return scaling_dict

def mse(input, target):
    #import pdb; pdb.set_trace()
    assert input.shape == target.shape, f"Input({input.shape}) and Target ({target.shape}) must have the same shape"
    return tf.math.reduce_mean(tf.square(input - target))



def mae(input, target, weight = None):
    if weight == None:
        return tf.math.reduce_mean(tf.math.abs(input - target))
    return tf.math.reduce_mean(weight * (tf.math.abs(input - target)))

def inv_scale_tf(scaling_dict, field, field_name):
    mean = tf.constant(scaling_dict[field_name][0], dtype = "float64")
    std = tf.constant(scaling_dict[field_name][1], dtype = "float64")
    scaled_field = tf.add(tf.multiply(field, std), mean)
    return scaled_field

# def generate_gnn_model(network_params):
#     return GraphNet(network_params)

def get_learning_rate(train_params):
    scheduler_name = 'exponential'
    if scheduler_name == 'exponential':
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = train_params['learning_rate'], decay_steps=1000, decay_rate=train_params['lr_decay'])
    elif scheduler_name == 'cosine':
        eta_min = train_params['learning_rate'] * train_params['lr_decay']
        scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = train_params['learning_rate'], decay_steps=nepochs, alpha = eta_min)
    return scheduler

def get_optimizer(train_params, scheduler):
    if train_params["optimizer_name"] == 'adam':
        #optimizer = tf.keras.optimizers.Adam(learning_rate = train_params['learning_rate'])
        optimizer = tf.keras.optimizers.Adam(learning_rate = scheduler)
        #optimizer = tf.keras.optimizers.SGD(train_params['learning_rate'])
        #optimizer = tf.keras.optimizers.Adamax(train_params['learning_rate'])
    else:
        raise ValueError('Optimizer ' + optimizer_name + ' not implemented')
    return optimizer

def get_graph_data_loader(dataset, batch_size):
    graph_data_loader = []
    num_samples = len(dataset); num_batches = int(np.ceil(num_samples/batch_size))
    print(f"num batches: {num_batches}.  num graphs: {num_samples}.")
    indices_shuff = [i for i in range(num_samples)]
    random.shuffle(indices_shuff)
    for batch_ind in range(num_batches):
        try:
            batch_indices = indices_shuff[batch_size*batch_ind : batch_size*(batch_ind+1)]
        except:
            batch_indices = indices_shuff[batch_size*batch_ind :]
        graph_data_loader.append(dgl.batch([dataset[ind].to("/gpu:0") for ind in batch_indices]))
    return graph_data_loader

def get_master_tensors_steady(dataloader):
    g = dataloader[0]
    input1 = tf.concat((g.nodes["inlet"].data['inlet_features'],
                            g.nodes["outlet"].data['outlet_features'][::2,:],
                            g.nodes["outlet"].data['outlet_features'][1::2,:]), axis = 1)
    input2 = tf.concat((g.nodes["inlet"].data['inlet_features'],
                            g.nodes["outlet"].data['outlet_features'][1::2,:],
                            g.nodes["outlet"].data['outlet_features'][::2,:]), axis = 1)
    input = tf.concat((input1, input2), axis = 0)

    output1 = tf.cast(g.nodes['outlet'].data['outlet_coefs'], dtype=tf.float64)[::2,:]
    output2 = tf.cast(g.nodes['outlet'].data['outlet_coefs'], dtype=tf.float64)[1::2,:]
    output = tf.concat((output1, output2), axis = 0)

    flow1 = tf.cast(g.nodes['outlet'].data['outlet_flows'], dtype=tf.float64)[::2,:]
    flow2 = tf.cast(g.nodes['outlet'].data['outlet_flows'], dtype=tf.float64)[1::2,:]
    flow = tf.concat((flow1, flow2), axis = 0)

    dP1 = tf.cast(g.nodes['outlet'].data['outlet_dP'], dtype=tf.float64)[::2,:]
    dP2 = tf.cast(g.nodes['outlet'].data['outlet_dP'], dtype=tf.float64)[1::2,:]
    dP= tf.concat((dP1, dP2), axis = 0)
    #import pdb; pdb.set_trace()
    return (input, output, flow, dP)

def get_master_tensors(dataloader):
    g = dataloader[0]
    input1 = tf.cast(tf.concat((g.nodes["inlet"].data['inlet_features'],
                            g.nodes["outlet"].data['outlet_features'][::2,:],
                            g.nodes["outlet"].data['outlet_features'][1::2,:]), axis = 1), dtype=tf.float64)
    input2 = tf.cast(tf.concat((g.nodes["inlet"].data['inlet_features'],
                            g.nodes["outlet"].data['outlet_features'][1::2,:],
                            g.nodes["outlet"].data['outlet_features'][::2,:]), axis = 1), dtype=tf.float64)
    input = tf.cast(tf.concat((input1, input2), axis = 0), dtype=tf.float64)

    output1 = tf.cast(g.nodes['outlet'].data['outlet_coefs'], dtype=tf.float64)[::2,:]
    output2 = tf.cast(g.nodes['outlet'].data['outlet_coefs'], dtype=tf.float64)[1::2,:]
    output = tf.concat((output1, output2), axis = 0)

    flow1 = tf.cast(g.nodes['outlet'].data['unsteady_outlet_flows'], dtype=tf.float64)[::2,:]
    flow2 = tf.cast(g.nodes['outlet'].data['unsteady_outlet_flows'], dtype=tf.float64)[1::2,:]
    flow = tf.concat((flow1, flow2), axis = 0)

    flow_der1 = tf.cast(g.nodes['outlet'].data['unsteady_outlet_flow_ders'], dtype=tf.float64)[::2,:]
    flow_der2 = tf.cast(g.nodes['outlet'].data['unsteady_outlet_flow_ders'], dtype=tf.float64)[1::2,:]
    flow_der = tf.concat((flow_der1, flow_der2), axis = 0)

    dP1 = tf.cast(g.nodes['outlet'].data['unsteady_outlet_dP'], dtype=tf.float64)[::2,:]
    dP2 = tf.cast(g.nodes['outlet'].data['unsteady_outlet_dP'], dtype=tf.float64)[1::2,:]
    dP= tf.concat((dP1, dP2), axis = 0)
    #import pdb; pdb.set_trace()
    return (input, output, flow, flow_der, dP)

def get_noise(input_tensor, noise_level):

    noise = tf.random.normal(input_tensor.shape,
                                mean=0,
                                stddev=noise_level,
                                dtype = tf.float64)
    return noise

def get_batched_tensors(master_tensors, batch_size, noise_level):
    input_tensor_data_loader = []
    output_tensor_data_loader = []
    flow_tensor_data_loader = []
    flow_der_tensor_data_loader = []
    dP_tensor_data_loader = []
    num_samples = master_tensors[0].shape[0]; num_batches = int(np.ceil(num_samples/batch_size))
    #print(f"num batches: {num_batches}.  num graphs: {num_samples}.")
    indices_shuff = [int(i) for i in range(num_samples)]
    random.shuffle(indices_shuff)
    for batch_ind in range(num_batches):
        try:
            batch_indices = indices_shuff[batch_size*batch_ind : batch_size*(batch_ind+1)]
        except:
            batch_indices = indices_shuff[batch_size*batch_ind :]

        input_tensor_data_loader.append(tf.gather(master_tensors[0], batch_indices, axis = 0))
        output_tensor_data_loader.append(tf.gather(master_tensors[1] + get_noise(master_tensors[1], noise_level), batch_indices, axis = 0))
        flow_tensor_data_loader.append(tf.gather(master_tensors[2], batch_indices, axis = 0))
        flow_der_tensor_data_loader.append(tf.gather(master_tensors[3], batch_indices, axis = 0))
        dP_tensor_data_loader.append(tf.gather(master_tensors[4], batch_indices, axis = 0))

    return (input_tensor_data_loader, output_tensor_data_loader, flow_tensor_data_loader, flow_der_tensor_data_loader, dP_tensor_data_loader)

def get_batched_tensors_steady(master_tensors, batch_size, noise_level):
    input_tensor_data_loader = []
    output_tensor_data_loader = []
    flow_tensor_data_loader = []
    dP_tensor_data_loader = []
    num_samples = master_tensors[0].shape[0]; num_batches = int(np.ceil(num_samples/batch_size))
    #print(f"num batches: {num_batches}.  num graphs: {num_samples}.")
    indices_shuff = [int(i) for i in range(num_samples)]
    random.shuffle(indices_shuff)
    for batch_ind in range(num_batches):
        try:
            batch_indices = indices_shuff[batch_size*batch_ind : batch_size*(batch_ind+1)]
        except:
            batch_indices = indices_shuff[batch_size*batch_ind :]
        #import pdb; pdb.set_trace()
        # batch_indices2 = []
        # for batch_ind in batch_indices:
            #batch_indices2 += [2*batch_ind, 2*batch_ind+1]

        input_tensor_data_loader.append(tf.gather(master_tensors[0], batch_indices, axis = 0))
        output_tensor_data_loader.append(tf.gather(master_tensors[1] + get_noise(master_tensors[1], noise_level), batch_indices, axis = 0))
        flow_tensor_data_loader.append(tf.gather(master_tensors[2], batch_indices, axis = 0))
        dP_tensor_data_loader.append(tf.gather(master_tensors[3], batch_indices, axis = 0))

    return (input_tensor_data_loader, output_tensor_data_loader, flow_tensor_data_loader, dP_tensor_data_loader)
