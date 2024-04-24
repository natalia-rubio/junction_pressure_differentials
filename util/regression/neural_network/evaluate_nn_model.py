import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *
import dgl
import tensorflow as tf
tf.random.set_seed(0)

from util.regression.neural_network.graphnet_nn import GraphNet
from util.regression.neural_network.training_nn import *
from dgl.data.utils import load_graphs

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'font.size': 18})




anatomy = "Pulmo_rand"
seed = 0
num_geos = 129

model_name = "1_hl_52_lsmlp_0_0931_lr_0_008_lrd_1e-05_wd_bs_29_nepochs_300_seed_0_geos_127"
steady_model = tf.keras.models.load_model("results/models/neural_network/steady/"+model_name+"_steady", compile=True)
dataset_loc =f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/{set_type}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset"
validation_dataset = load_dict(dataset_loc)
unsteady = False
network_params = {
               'latent_size_mlp': 52, # 48,# 20,
               'out_size': 2,
               'process_iterations': 1,
               'hl_mlp':1,
               'num_inlet_ft' : 2,
               'num_outlet_ft': 3,
               'unsteady': unsteady,
               'output_name': "outlet_coefs"}


train_params = {'learning_rate': 0.0931,# 0.018, #0.1,
                'lr_decay': 0.00834, #0.031, #0.5,
                'batch_size': 29, #24,# int(np.ceil(len(train_dataset)/10)),
                'nepochs': 300,
                'weight_decay': 10**(-5),
                'optimizer_name' : "adam"}
scaling_dict = load_dict(f"data/scaling_dictionaries/Pulmo_rand_scaling_dict")
unsteady = network_params["unsteady"]

print('val dataset contains {:.0f} graphs'.format(len(validation_dataset)))

nepochs = train_params['nepochs']
learning_rate = get_learning_rate(train_params)
optimizer = get_optimizer(train_params, learning_rate)

validation_dataloader = get_graph_data_loader(validation_dataset, batch_size=len(validation_dataset))
if unsteady:
    validation_master_tensors = get_master_tensors_unsteady(validation_dataloader)
else:
    validation_master_tensors = get_master_tensors_steady(validation_dataloader)

input_tensor = validation_master_tensors[0]
validation_output_tensor_data_loader = validation_master_tensors[1]
flow_tensor = validation_master_tensors[2]
validation_flow_der_tensor_data_loader = validation_master_tensors[3]
validation_dP_tensor_data_loader = validation_master_tensors[4]

pred_dP_steady = tf.reshape(inv_scale_tf(scaling_dict, steady_model(input_tensor)[:,0], "coef_a"), (-1,1)) * tf.square(flow_tensor) + \
                tf.reshape(inv_scale_tf(scaling_dict, steady_model(input_tensor)[:,1], "coef_b"), (-1,1)) * flow_tensor

import pdb; pdb.set_trace()
