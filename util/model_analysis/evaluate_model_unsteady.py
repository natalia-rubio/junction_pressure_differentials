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

def plot_unsteady(anatomy):

    model_name = f"1_hl_30_lsmlp_0_02_lr_0_05_lrd_1e-05_wd_bs_2_nepochs_200_seed_2_geos_45"
    nn_model = tf.keras.models.load_model("results/models/neural_network/unsteady/"+model_name+"/nn_model", compile=True)
    graph_list = load_dict("data/graph_lists/val_aorta_num_geos_45_seed_2_graph_list")
    graph = graph_list[0]
    scaling_dict = load_dict("data/scaling_dictionaries/aorta_scaling_dict_unsteady")

    master_tensor = get_master_tensors([graph])

    input_tensor = master_tensor[0]
    output_tensor = master_tensor[1][0,:]
    flow_tensor = master_tensor[2][0,:]
    flow_der_tensor = master_tensor[3][0,:]
    dP_tensor = master_tensor[4][0,:]

    pred_outlet_coefs = tf.cast(nn_model.predict(input_tensor), dtype=tf.float32)
    pred_dP = tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,0], "coef_a"), (-1,1)) * tf.square(flow_tensor) + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,1], "coef_b"), (-1,1)) * flow_tensor + \
                tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,2], "coef_L"), (-1,1)) * (flow_der_tensor)

    # # Apply Unified0D Method
    junction_dict_global = graphs_to_junction_dict([graph_list[0]], scaling_dict)
    flow_arr = flow_tensor.numpy()
    dP_mynard_list = []

    for i in range(flow_arr.size):
        #junction_dict["outlet_velocity"] = flow_arr[i]
        dP_mynard_list = dP_mynard_list + [-1*apply_unified0D_plus(junction_dict_global[i])[0]]
    dP_mynard = np.asarray(dP_mynard_list)/1333

    plt.clf()
    plt.plot(np.asarray(flow_tensor), np.asarray(tf.reshape(pred_dP[0,:], [-1,]))/1333, label = 'NN', c = "royalblue", linewidth=2)
    plt.plot(np.asarray(flow_tensor), dP_mynard, label = 'Unified0D+', c = "salmon", linewidth=2, linestyle ="--")
    plt.scatter(np.asarray(flow_tensor), np.asarray(dP_tensor)/1333, label = "Simulation", c = "peru", marker = "*", s = 100)

    plt.xlabel("$Q \;  (\mathrm{cm^3/s})$")
    plt.ylabel("$\Delta P$ (mmHg)")
    plt.legend(fontsize="14")
    plt.savefig(f"results/model_visualization/unsteady_flow_vs_predicted_dps.pdf", bbox_inches='tight', transparent=True, format = "pdf")
    #import pdb; pdb.set_trace()

    return

plot_unsteady(anatomy = "Aorta_u_40-60_over15_val")
