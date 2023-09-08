import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *
from util.tools.junction_proc import *

def assemble_graphs_steady(anatomy):

    graph_list = []

    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict_steady")
    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_scaling_dict_steady")

    for i in range(int(len(char_val_dict["inlet_radius"])/2)):

        inlet_data = np.stack((scale(scaling_dict, np.asarray(char_val_dict["inlet_area"][2*i]), "inlet_area").reshape(1,-1),
                                )).T

        outlet_data = np.stack((scale(scaling_dict, np.asarray(char_val_dict["outlet_area"][2*i: 2*(i+1)]), "outlet_area"),
                                scale(scaling_dict, np.asarray(char_val_dict["angle"][2*i: 2*(i+1)]), "angle")
                                )).T

        # outlet_data = np.asarray([scale(scaling_dict, char_val_dict["coef_a"][2*i: 2*(i+1)], "coef_a"),
        #                         scale(scaling_dict, char_val_dict["coef_b"][2*i: 2*(i+1)], "coef_b")]).T

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
            graph.nodes["inlet"].data["inlet_features"] = tf.reshape(tf.convert_to_tensor(inlet_data, dtype=tf.float64), [1,1])
            graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float64)
            graph.nodes["outlet"].data["outlet_flows"] = tf.convert_to_tensor(outlet_flows, dtype=tf.float64)
            graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_dPs, dtype=tf.float64)
            graph.nodes["inlet"].data["geo_name"] = tf.constant([geo_name])
            graph.nodes["outlet"].data["outlet_coefs"] = tf.convert_to_tensor(outlet_coefs, dtype=tf.float64)

        graph_list.append(graph)

    dgl.save_graphs(f"data/graph_lists/{anatomy}_graph_list_steady", graph_list)
    return graph
