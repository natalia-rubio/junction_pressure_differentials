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

def vary_param(anatomy, variable, dP_type):
    plt.clf()
    #model_name = "1_hl_20_lsmlp_0_02_lr_0_7_lrd_1e-05_wd_bs_8_nepochs_200_seed_0_geos_187"#mynard
    model_name = "1_hl_20_lsmlp_0_02_lr_0_7_lrd_1e-05_wd_bs_5_nepochs_200_seed_0_geos_110"#aorta
    nn_model = tf.keras.models.load_model("results/models/neural_network/steady/"+model_name, compile=True)

    #junction_params = load_dict(f"/home/nrubio/Desktop/synthetic_junctions/Aorta_vary_r2/{geo}/junction_params_dict")
    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_vary_rout_synthetic_data_dict_steady")
    #char_val_dict0 = load_dict(f"data/characteristic_value_dictionaries/Aorta_vary_rout/synthetic_data_dict_steady")
    #print(char_val_dict)
    #scaling_dict = load_dict(f"data/scaling_dictionaries/mynard_rand_scaling_dict_steady")
    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_rand_scaling_dict_steady")
    print(scaling_dict)
    #scaling_dict = load_dict(f"data/scaling_dictionaries/Aorta_u_40-60_over3_scaling_dict_steady")
    dPs = []
    print(char_val_dict)
    for i in range(int(len(char_val_dict["name"])/2)):

        if i/int(len(char_val_dict["name"])/2) < 0.5:
            outlet_ind = 1
        else:
            outlet_ind = 0
        print(outlet_ind)

        inlet_data = np.stack((scale(scaling_dict, char_val_dict["inlet_area"][2*i], "inlet_area").reshape(1,-1),
                                )).T

        outlet_data = np.stack((
            scale(scaling_dict, np.asarray(char_val_dict["outlet_area"][2*i: 2*(i+1)]), "outlet_area"),
            scale(scaling_dict, np.asarray(char_val_dict["angle"][2*i: 2*(i+1)]), "angle"),
            )).T
        #print(outlet_data)
        outlet_flows = np.stack((np.asarray(char_val_dict["flow_list"][2*i]).T,
                                np.asarray(char_val_dict["flow_list"][2*i + 1]).T))

        outlet_dPs = np.stack((np.asarray(char_val_dict["dP_list"][2*i]).T,
                                np.asarray(char_val_dict["dP_list"][2*i + 1]).T))

        outlet_junction_dPs = np.stack((np.asarray(char_val_dict["dP_junc_list"][2*i]).T,
                                np.asarray(char_val_dict["dP_junc_list"][2*i + 1]).T))

        outlet_coefs = np.asarray([scale(scaling_dict, char_val_dict["coef_a"][2*i: 2*(i+1)], "coef_a"),
                                scale(scaling_dict, char_val_dict["coef_b"][2*i: 2*(i+1)], "coef_b")]).T

        geo_name = "".join([let for let in char_val_dict["name"][2*i] if let.isnumeric()])
        geo_name = int(geo_name)

        inlet_outlet_pairs = get_inlet_outlet_pairs(1, 2)
        outlet_pairs = get_outlet_pairs(2)
        graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
        graph  = graph.to("/cpu:0")

        with tf.device("/cpu:0"):

            graph.nodes["inlet"].data["inlet_features"] = tf.reshape(tf.convert_to_tensor(inlet_data, dtype=tf.float64), [1,1])
            graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float64)
            graph.nodes["outlet"].data["outlet_flows"] = tf.convert_to_tensor(outlet_flows, dtype=tf.float64)
            if dP_type == "end":
                graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_dPs, dtype=tf.float64)
            elif dP_type == "junction":
                graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_junction_dPs, dtype=tf.float64)
            graph.nodes["inlet"].data["geo_name"] = tf.constant([geo_name])
            graph.nodes["outlet"].data["outlet_coefs"] = tf.convert_to_tensor(outlet_coefs, dtype=tf.float64)
        print(graph.nodes["outlet"].data)

        master_tensor = get_master_tensors_steady([graph])
        input_tensor = master_tensor[0]
        flow_tensor = master_tensor[2]
        dP = master_tensor[3]

        flow_tensor_cont = tf.linspace(flow_tensor[outlet_ind,0], flow_tensor[outlet_ind,-1], 100)
        inflow_tensor_cont =  tf.linspace(flow_tensor[0,0], flow_tensor[0,-1], 100) \
                            + tf.linspace(flow_tensor[1,0], flow_tensor[1,-1], 100)

        pred_outlet_coefs = tf.cast(nn_model.predict(input_tensor), dtype=tf.float64)

        pred_dP = tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[outlet_ind,0], "coef_a"), (-1,1)) * tf.square(flow_tensor_cont) + \
                    tf.reshape(inv_scale_tf(scaling_dict, pred_outlet_coefs[outlet_ind,1], "coef_b"), (-1,1)) * flow_tensor_cont

        dPs.append(np.asarray(pred_dP))

        junction_dict_global = graphs_to_junction_dict_steady([graph], scaling_dict)
        flow_arr = flow_tensor.numpy()
        dP_mynard_list = []
        if dP_type == "end":
            for j in range(1,100):
                    dP_mynard_list = dP_mynard_list + [apply_unified0D_plus(junction_dict_global[j])[outlet_ind] \
                                    - dP_poiseuille(flow = inflow_tensor_cont[j], radius = char_val_dict["inlet_radius"][2*i], length = char_val_dict["inlet_length"][2*i]) \
                                    - dP_poiseuille(flow = flow_tensor_cont[j], radius = char_val_dict["outlet_radius"][2*i+outlet_ind], length = char_val_dict["outlet_length"][2*i+outlet_ind])]
        elif dP_type == "junction":
            for j in range(1,100):
                    dP_mynard_list = dP_mynard_list + [apply_unified0D_plus(junction_dict_global[j])[outlet_ind]]
        dP_mynard = np.asarray(dP_mynard_list)

        if variable == "rout":
            plt.plot(np.asarray(flow_tensor_cont), np.asarray(tf.reshape(pred_dP, [-1,]))/1333, label = f"outlet radius = { char_val_dict['outlet_radius'][2*i+outlet_ind]:.2f} (cm)", c = colors[i], linewidth=2)
            
            plt.plot(np.asarray(flow_tensor_cont)[1:], dP_mynard/1333, "--", c = colors[i], linewidth=2 )
            #plt.plot(np.asarray(flow_tensor_cont)[1:], dP_mynard/1333, "--", c = colors[i], linewidth=2, label = f"unified0D_plus")

        if variable == "angle":
            plt.plot(np.asarray(flow_tensor_cont), np.asarray(tf.reshape(pred_dP, [-1,]))/1333, label = f"Outlet Angle = {char_val_dict['angle'][2*i+1]:.2f} $^\circ$", c = colors[i])

        U_in = np.asarray(flow_tensor[0,1:] + flow_tensor[1,1:])
        U_out = np.asarray(flow_tensor)[outlet_ind,1:]
        K = -(np.asarray(dP)[outlet_ind,1:] - (0.5*1.06*(U_in**2 - U_out**2)))/ (0.5* 1.06 * U_in**2)

        plt.scatter(np.asarray(flow_tensor)[outlet_ind,:], np.asarray(dP)[outlet_ind,:]/1333, c = colors[i], marker = "*", s = 100)
        #plt.scatter(np.asarray(flow_tensor)[outlet_ind,1:], K, c = colors[i], marker = "*", s = 100)

    plt.xlabel("$Q \;  (\mathrm{cm^3/s})$")
    plt.ylabel("$\Delta P$ (mmHg)")
    plt.legend(fontsize="14")
    plt.savefig(f"results/model_visualization/{anatomy}_{variable}_vs_predicted_dps_{dP_type}_steady.pdf", bbox_inches='tight', format = "pdf")
    return


# vary_param("Aorta_vary_rout_over_5", "rout", dP_type = "junction")
#
# vary_param("Aorta_vary_rout_over_5", "rout", dP_type = "end")
# vary_param("mynard_vary_rout", "rout", dP_type = "junction")
#
# vary_param("mynard_vary_rout", "rout", dP_type = "end")
#

anatomy = sys.argv[1];
vary_param(anatomy, "rout", "end")
