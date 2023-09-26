import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.unified0D_plus.apply_unified0D_plus  import *
from util.unified0D_plus.graph_to_junction_dict import *

from util.regression.neural_network.training_util import *
from util.tools.graph_handling import *
from util.tools.basic import *
import tensorflow as tf
import dgl
from dgl.data import DGLDataset

def dP_poiseuille(flow, radius, length):
    mu = 0.04
    dP = 8 * mu * length * flow /(np.pi * radius**4)
    return dP

def plot_unsteady(anatomy):

    model_name = f"1_hl_20_lsmlp_0_02_lr_0_7_lrd_1e-05_wd_bs_5_nepochs_200_seed_0_geos_102"
    nn_model = tf.keras.models.load_model("results/models/neural_network/steady/"+model_name, compile=True)
    graph_list = load_dict(f"data/graph_lists/{anatomy}/val_Aorta_rand_num_geos_102_seed_0_graph_list")
    graph = graph_list[19]
    scaling_dict = load_dict("data/scaling_dictionaries/Aorta_rand_scaling_dict")

    master_tensor = get_master_tensors_unsteady([graph])

    input_tensor = master_tensor[0]
    output_tensor = master_tensor[1][0,:]
    flow_tensor = master_tensor[2][0,:]
    flow_der_tensor = master_tensor[3][0,:]
    dP_tensor = master_tensor[4][0,:]

    print(f"Flow tensor: {flow_tensor}")
    print(f"Flow derivative tensor: {flow_der_tensor}")
    print(f"dP derivative tensor: {dP_tensor}")

    pred_outlet_coefs = tf.cast(nn_model.predict(input_tensor), dtype=tf.float64)
    pred_dP = tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,0], "coef_a"), (-1,1)) * tf.square(flow_tensor) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,1], "coef_b"), (-1,1)) * flow_tensor + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,2], "coef_L"), (-1,1)) * (flow_der_tensor)

    pred_dP_steady  = tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,0], "coef_a"), (-1,1)) * tf.square(flow_tensor) + \
                    tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,1], "coef_b"), (-1,1)) * flow_tensor
    # Apply Unified0D Method
    junction_dict_global = graphs_to_junction_dict([graph_list[0]], scaling_dict)

    flow_arr = flow_tensor.numpy()
    dP_mynard_list = []

    outlet_ind = 0
    for i in range(flow_arr.size):
        dP_mynard_list = dP_mynard_list + [apply_unified0D_plus(junction_dict_global[i])[outlet_ind] \
                        - dP_poiseuille(flow = flow_arr[i], radius = junction_dict_global[i]["inlet_radius"][0], length = junction_dict_global[i]["inlet_length"][outlet_ind]) \
                        - dP_poiseuille(flow = flow_arr[i], radius = junction_dict_global[i]["outlet_radius"][outlet_ind], length = junction_dict_global[i]["outlet_length"][outlet_ind])]
    dP_mynard = np.asarray(dP_mynard_list)/1333

    plt.clf()
    plt.plot(np.asarray(flow_tensor), np.asarray(tf.reshape(pred_dP[0,:], [-1,]))/1333, label = 'RRI (NN)', c = "royalblue", linewidth=2)
    plt.plot(np.asarray(flow_tensor), np.asarray(tf.reshape(pred_dP_steady[0,:], [-1,]))/1333, label = 'RR (NN)', c = "seagreen", linewidth=2)
    plt.plot(np.asarray(flow_tensor), dP_mynard, label = 'Unified0D+', c = "salmon", linewidth=2, linestyle ="--")
    plt.scatter(np.asarray(flow_tensor), np.asarray(dP_tensor)/1333, label = "Simulation", c = "peru", marker = "*", s = 100)

    plt.xlabel("$Q \;  (\mathrm{cm^3/s})$")
    plt.ylabel("$\Delta P$ (mmHg)")
    plt.legend(fontsize="14")
    plt.savefig(f"results/model_visualization/unsteady_flow_vs_predicted_dps.pdf", bbox_inches='tight', transparent=True, format = "pdf")
    import pdb; pdb.set_trace()

    return

plot_unsteady(anatomy = "Aorta_rand")
