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

    pulmo_model_name = "1_hl_52_lsmlp_0_0931_lr_0_008_lrd_1e-05_wd_bs_29_nepochs_300_seed_0_geos_127"
    aorta_model_name = "1_hl_52_lsmlp_0_0931_lr_0_008_lrd_1e-05_wd_bs_29_nepochs_300_seed_0_geos_110"

    pulmo_graph_list_name = f"data/graph_lists/{anatomy}_rand/val_Pulmo_rand_num_geos_127_seed_0_graph_list"
    aorta_graph_list_name = f"data/graph_lists/{anatomy}_rand/val_Aorta_rand_num_geos_110_seed_0_graph_list"

    if anatomy == "Pulmo":
        model_name = pulmo_model_name
        graph_list_name = pulmo_graph_list_name
        geo_list = [25, 9, 7, 4, 24] #[16,9,8,1,24]#
        rmse_list = [0.29, 0.51, 0.70, 0.88, 2.77]
    elif anatomy == "Aorta":
        model_name = aorta_model_name
        graph_list_name = aorta_graph_list_name
        geo_list = [2, 20, 12, 9, 15]
        rmse_list = [0.90, 1.1, 1.3, 1.6, 7.4]

    nn_model = tf.keras.models.load_model("results/models/neural_network/steady/"+model_name, compile=True)
    graph_list = load_dict(graph_list_name)
    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_rand_scaling_dict")


    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize = (4, 12), sharex = True, sharey = True, layout = "constrained")
    for i in range(len(geo_list)):
        ax = axs.flat[i]
        graph = graph_list[geo_list[i]]
        master_tensor = get_master_tensors_unsteady([graph])

        input_tensor = master_tensor[0]
        output_tensor = master_tensor[1][0,:]
        flow_tensor = master_tensor[2][0,:]
        flow_der_tensor = master_tensor[3][0,:]
        dP_tensor = master_tensor[4][0,:]

        pred_outlet_coefs = tf.cast(nn_model.predict(input_tensor), dtype=tf.float64)
        pred_dP = tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,0], "coef_a"), (-1,1)) * tf.square(flow_tensor) + \
                    tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,1], "coef_b"), (-1,1)) * flow_tensor + \
                    tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,2], "coef_L"), (-1,1)) * (flow_der_tensor)

        pred_dP_steady  = tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,0], "coef_a"), (-1,1)) * tf.square(flow_tensor) + \
                        tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[:,1], "coef_b"), (-1,1)) * flow_tensor


        ax.plot(np.asarray(flow_tensor), np.asarray(tf.reshape(pred_dP[0,:], [-1,]))/1333, label = 'RRI (NN)', c = "royalblue", linewidth=2)
        ax.plot(np.asarray(flow_tensor), np.asarray(tf.reshape(pred_dP_steady[0,:], [-1,]))/1333, label = 'RR (NN)', c = "seagreen", linewidth=2)
        ax.text(0.01, 0.02, f"RMSE {rmse_list[i]} mmHg", transform=ax.transAxes)
        # plt.plot(np.asarray(flow_tensor), dP_mynard, label = 'Unified0D+', c = "salmon", linewidth=2, linestyle ="--")
        ax.scatter(np.asarray(flow_tensor), np.asarray(dP_tensor)/1333, label = "Simulation", c = "peru", marker = "*", s = 100)
        #ax.set_title(f"Test Geometry with RMSE {rmse_list[i]} mmHg")
        # ax.set_ylabel("$\Delta P$ (mmHg)")

        # if i == 4:
        #     ax.set_xlabel("$Q \;  (\mathrm{cm^3/s})$")
        if i == 0:
            ax.legend(fontsize="14", loc = "upper right")
    fig.text(0.5, -0.02, "$\Delta P$ (mmHg)", ha = "center")
    fig.text(-0.08, 0.5, "$Q \;  (\mathrm{cm^3/s})$", rotation = "vertical", va = "center")
    fig.savefig(f"results/model_visualization/{anatomy}_unsteady_flow_vs_predicted_dps.pdf", bbox_inches='tight', transparent=True, format = "pdf")

    return

anatomy = sys.argv[1]
plot_unsteady(anatomy = anatomy)
#
# junction_dict_global = graphs_to_junction_dict([graph_list[0]], scaling_dict)
#
# flow_arr = flow_tensor.numpy()
# dP_mynard_list = []
#
# outlet_ind = 0
# for i in range(flow_arr.size):
#     dP_mynard_list = dP_mynard_list + [apply_unified0D_plus(junction_dict_global[i])[outlet_ind] \
#                     - dP_poiseuille(flow = flow_arr[i], radius = junction_dict_global[i]["inlet_radius"][0], length = junction_dict_global[i]["inlet_length"][outlet_ind]) \
#                     - dP_poiseuille(flow = flow_arr[i], radius = junction_dict_global[i]["outlet_radius"][outlet_ind], length = junction_dict_global[i]["outlet_length"][outlet_ind])]
# dP_mynard = np.asarray(dP_mynard_list)/1333
