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

def vary_param(anatomy, variable):

    #model_name = "1_hl_40_lsmlp_0_02_lr_0_05_lrd_1e-05_wd_bs_15_nepochs_200_seed_2_geos_360"#"1_hl_30_lsmlp_0_02_lr_0_05_lrd_1e-05_wd_bs_15_nepochs_200_seed_2_geos_360"
    #nn_model = tf.keras.models.load_model("results/models/steady/"+model_name+"/nn_model", compile=True)

    #junction_params = load_dict(f"/home/nrubio/Desktop/synthetic_junctions/Aorta_vary_r2/{geo}/junction_params_dict")

    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/mynard_vary_mesh_synthetic_data_dict_steady")
    scaling_dict = load_dict("data/scaling_dictionaries/mynard_vary_mesh_scaling_dict_steady")
    dPs = []
    marker_list = ["o", "v", "d", "*", "X"]
    ref_list = ["4.0E4", "1.0E3", "6.3E4", "2.1E5", "9.7E5"]
    for i in range(int(len(char_val_dict["name"])/2)):

        inlet_data = np.stack((scale(scaling_dict, np.asarray(char_val_dict["inlet_radius"][2*i]), "inlet_radius").reshape(1,-1),
                                )).T

        outlet_data = np.stack((
            scale(scaling_dict, np.asarray(char_val_dict["outlet_radius"][2*i: 2*(i+1)]), "outlet_radius"),
            scale(scaling_dict, np.asarray(char_val_dict["angle"][2*i: 2*(i+1)]), "angle")
            )).T
        #print(outlet_data)
        outlet_flows = np.stack((np.asarray(char_val_dict["flow_list"][2*i]).T,
                                np.asarray(char_val_dict["flow_list"][2*i + 1]).T))

        outlet_dPs = np.stack((np.asarray(char_val_dict["dP_list"][2*i]).T,
                                np.asarray(char_val_dict["dP_list"][2*i + 1]).T))

        outlet_coefs = np.asarray([scale(scaling_dict, char_val_dict["coef_a"][2*i: 2*(i+1)], "coef_a"),
                                scale(scaling_dict, char_val_dict["coef_b"][2*i: 2*(i+1)], "coef_b")]).T

        geo_name = "".join([let for let in char_val_dict["name"][2*i] if let.isnumeric()])
        geo_name = int(geo_name)

        inlet_outlet_pairs = get_inlet_outlet_pairs(1, 2)
        outlet_pairs = get_outlet_pairs(2)
        graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
        graph  = graph.to("/cpu:0")

        with tf.device("/cpu:0"):

            graph.nodes["inlet"].data["inlet_features"] = tf.reshape(tf.convert_to_tensor(inlet_data, dtype=tf.float32), [1,1])
            graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
            graph.nodes["outlet"].data["outlet_flows"] = tf.convert_to_tensor(outlet_flows, dtype=tf.float32)
            graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_dPs, dtype=tf.float32)
            graph.nodes["inlet"].data["geo_name"] = tf.constant([geo_name])
            graph.nodes["outlet"].data["outlet_coefs"] = tf.convert_to_tensor(outlet_coefs, dtype=tf.float32)

        master_tensor = get_master_tensors_steady([graph])
        input_tensor = master_tensor[0]; output = master_tensor[1]; flow_tensor = master_tensor[2]; dP = master_tensor[3]

        flow_tensor_cont = tf.linspace(flow_tensor[1,0], flow_tensor[1,-1], 100)
        plt.scatter(np.asarray(flow_tensor)[0,:], np.asarray(dP)[0,:]/1333, facecolors='none', edgecolors = "royalblue", marker = marker_list[i], s = 100, label = f"{ref_list[i]} mesh points")


    junction_dict_global = graphs_to_junction_dict_steady([graph], scaling_dict)
    flow_arr = flow_tensor.numpy()
    dP_mynard_list = []
    for j in range(1,100):
        dP_mynard_list = dP_mynard_list + [apply_unified0D_plus(junction_dict_global[j])[0]]
    dP_mynard = np.asarray(dP_mynard_list)/1333

    plt.plot(np.asarray(flow_tensor_cont)[1:], dP_mynard, "--", c = "royalblue", label = "Unified0D+", linewidth=2)

    plt.xlabel("$Q \;  (\mathrm{cm^3/s})$")
    plt.ylabel("$\Delta P$ (mmHg)")
    plt.legend(fontsize="12", bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(f"results/unified0D_plus/mesh_refinement_study.pdf", bbox_inches='tight', format = "pdf")
    return

vary_param("Aorta_vary_rout", "rout")
#vary_param("Aorta_vary_angle", "angle")
