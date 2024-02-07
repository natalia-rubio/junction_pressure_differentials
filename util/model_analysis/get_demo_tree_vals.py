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

def get_re(flow, radius):
    u = flow/(np.pi * radius**2)
    d = 2*radius
    rho = 1.06
    mu = 0.04
    return rho*u*d/mu

def dP_poiseuille(flow, radius, length):
    mu = 0.04
    dP = 8 * mu * length * flow /(np.pi * radius**4)
    return dP

anatomy = "Aorta"
model_name = "1_hl_52_lsmlp_0_0931_lr_0_008_lrd_1e-05_wd_bs_29_nepochs_300_seed_0_geos_110_dP"#aorta
scaling_dict = load_dict(f"data/scaling_dictionaries/Aorta_rand_scaling_dict")
#import pdb; pdb.set_trace()
    # elif anatomy == "Pulmo":
    #     model_name = "1_hl_52_lsmlp_0_0931_lr_0_008_lrd_1e-05_wd_bs_29_nepochs_300_seed_0_geos_123_dP"
    #     char_val_dict = load_dict(f"data/characteristic_value_dictionaries/Pulmo_vary_rout_synthetic_data_dict")
    #     scaling_dict = load_dict(f"data/scaling_dictionaries/Pulmo_rand_scaling_dict")
node_data_dict = load_dict(f"data/synthetic_tree/node_data_dict")
nn_model_steady = tf.keras.models.load_model("results/models/neural_network/steady/"+model_name+"_steady", compile=True)
nn_model_unsteady = tf.keras.models.load_model("results/models/neural_network/steady/"+model_name, compile=True)
    #print(nn_model.get_config())
inlet_data = np.stack((scale(scaling_dict, node_data_dict[0]["area"], "inlet_area").reshape(1,-1),
                                scale(scaling_dict, node_data_dict[0]["dist_to_bif"], "inlet_length").reshape(1,-1),
                                )).T

outlet_data = np.stack((
            scale(scaling_dict, np.asarray([node_data_dict[1]["area"], node_data_dict[2]["area"]]), "outlet_area"),
            scale(scaling_dict, np.asarray([node_data_dict[1]["dist_to_bif"], node_data_dict[2]["dist_to_bif"]]), "outlet_length"),
            scale(scaling_dict, np.asarray([20, 20]), "angle"))).T
#print(outlet_data)
outlet_flows = np.stack((np.asarray([0,0,0,0,0]).T,
                        np.asarray([0,0,0,0,0]).T))

outlet_dPs = np.stack((np.asarray([0,0,0,0,0]).T,
                        np.asarray([0,0,0,0,0]).T))

outlet_junction_dPs = np.stack((np.asarray([0,0,0,0,0]).T,
                        np.asarray([0,0,0,0,0]).T))

outlet_coefs = np.asarray([scale(scaling_dict, np.asarray([0,0]), "coef_a"),
                        scale(scaling_dict, np.asarray([0,0]), "coef_b"),
                        scale(scaling_dict, np.asarray([0,0]), "coef_L")]).T

geo_name = 1
geo_name = int(geo_name)

inlet_outlet_pairs = get_inlet_outlet_pairs(1, 2)
outlet_pairs = get_outlet_pairs(2)
graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
graph  = graph.to("/cpu:0")

with tf.device("/cpu:0"):

    graph.nodes["inlet"].data["inlet_features"] = tf.reshape(tf.convert_to_tensor(inlet_data, dtype=tf.float64), [1,-1])
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float64)
    graph.nodes["outlet"].data["outlet_flows"] = tf.convert_to_tensor(outlet_flows, dtype=tf.float64)
    graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_dPs, dtype=tf.float64)

    graph.nodes["inlet"].data["geo_name"] = tf.constant([geo_name])
    graph.nodes["outlet"].data["outlet_coefs"] = tf.convert_to_tensor(outlet_coefs, dtype=tf.float64)

master_tensor = get_master_tensors_steady([graph])
input_tensor = master_tensor[0]; print(input_tensor)
pred_outlet_coefs_steady = tf.cast(nn_model_steady.predict(input_tensor), dtype=tf.float64)
a = inv_scale_tf(scaling_dict, pred_outlet_coefs_steady[:,0], "coef_a").numpy()
b = inv_scale_tf(scaling_dict, pred_outlet_coefs_steady[:,1], "coef_b").numpy()
pred_outlet_coefs_unsteady = tf.cast(nn_model_unsteady.predict(input_tensor), dtype=tf.float64)
L = inv_scale_tf(scaling_dict, pred_outlet_coefs_unsteady[:,0], "coef_L").numpy()
print(f"Aorta: {a}, {b}, {L}")

aorta_coef_list = [a, b, L]

anatomy = "Pulmo"
model_name = "1_hl_52_lsmlp_0_0931_lr_0_008_lrd_1e-05_wd_bs_29_nepochs_300_seed_0_geos_123_dP"
scaling_dict = load_dict(f"data/scaling_dictionaries/Pulmo_rand_scaling_dict")
#import pdb; pdb.set_trace()
    # elif anatomy == "Pulmo":
    #     model_name = "1_hl_52_lsmlp_0_0931_lr_0_008_lrd_1e-05_wd_bs_29_nepochs_300_seed_0_geos_123_dP"
    #     char_val_dict = load_dict(f"data/characteristic_value_dictionaries/Pulmo_vary_rout_synthetic_data_dict")
    #     scaling_dict = load_dict(f"data/scaling_dictionaries/Pulmo_rand_scaling_dict")
node_data_dict = load_dict(f"data/synthetic_tree/node_data_dict")
nn_model_steady = tf.keras.models.load_model("results/models/neural_network/steady/"+model_name+"_steady", compile=True)
nn_model_unsteady = tf.keras.models.load_model("results/models/neural_network/steady/"+model_name, compile=True)
    #print(nn_model.get_config())
inlet_data = np.stack((scale(scaling_dict, node_data_dict[3]["area"], "inlet_area").reshape(1,-1),
                                scale(scaling_dict, node_data_dict[3]["dist_to_bif"], "inlet_length").reshape(1,-1),
                                )).T

outlet_data = np.stack((
            scale(scaling_dict, np.asarray([node_data_dict[4]["area"], node_data_dict[5]["area"]]), "outlet_area"),
            scale(scaling_dict, np.asarray([node_data_dict[4]["dist_to_bif"], node_data_dict[5]["dist_to_bif"]]), "outlet_length"),
            scale(scaling_dict, np.asarray([16, 16]), "angle"))).T
#print(outlet_data)
outlet_flows = np.stack((np.asarray([0,0,0,0,0]).T,
                        np.asarray([0,0,0,0,0]).T))

outlet_dPs = np.stack((np.asarray([0,0,0,0,0]).T,
                        np.asarray([0,0,0,0,0]).T))

outlet_junction_dPs = np.stack((np.asarray([0,0,0,0,0]).T,
                        np.asarray([0,0,0,0,0]).T))

outlet_coefs = np.asarray([scale(scaling_dict, np.asarray([0,0]), "coef_a"),
                        scale(scaling_dict, np.asarray([0,0]), "coef_b"),
                        scale(scaling_dict, np.asarray([0,0]), "coef_L")]).T

geo_name = 2
geo_name = int(geo_name)

inlet_outlet_pairs = get_inlet_outlet_pairs(1, 2)
outlet_pairs = get_outlet_pairs(2)
graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
graph  = graph.to("/cpu:0")

with tf.device("/cpu:0"):

    graph.nodes["inlet"].data["inlet_features"] = tf.reshape(tf.convert_to_tensor(inlet_data, dtype=tf.float64), [1,-1])
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float64)
    graph.nodes["outlet"].data["outlet_flows"] = tf.convert_to_tensor(outlet_flows, dtype=tf.float64)
    graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_dPs, dtype=tf.float64)

    graph.nodes["inlet"].data["geo_name"] = tf.constant([geo_name])
    graph.nodes["outlet"].data["outlet_coefs"] = tf.convert_to_tensor(outlet_coefs, dtype=tf.float64)

master_tensor = get_master_tensors_steady([graph])
input_tensor = master_tensor[0]; print(input_tensor)
pred_outlet_coefs_steady = tf.cast(nn_model_steady.predict(input_tensor), dtype=tf.float64)
a = inv_scale_tf(scaling_dict, pred_outlet_coefs_steady[:,0], "coef_a").numpy()
b = inv_scale_tf(scaling_dict, pred_outlet_coefs_steady[:,1], "coef_b").numpy()
pred_outlet_coefs_unsteady = tf.cast(nn_model_unsteady.predict(input_tensor), dtype=tf.float64)
L = inv_scale_tf(scaling_dict, pred_outlet_coefs_unsteady[:,0], "coef_L").numpy()
print(f"Pulmo: {a}, {b}, {L}")

pulmo_coef_list = [a, b, L]

coef_dict = {"Aorta": aorta_coef_list,
    "Pulmo": pulmo_coef_list}

save_dict(coef_dict, "data/synthetic_tree/coef_dict")
