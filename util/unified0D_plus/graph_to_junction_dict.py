import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *

def graphs_to_junction_dict(graph_list, scaling_dict):

    junction_dict_global = {}
    junction_counter = 0 # count total number of junctions

    for graph in graph_list: # loop over all data points
        num_time_steps = (graph.nodes["outlet"].data["unsteady_outlet_flows"]).numpy().shape[1]
        for time_ind in range(num_time_steps):
            junction_dict_global.update( # add junction if missing
                    {junction_counter: get_features_from_graph(graph, scaling_dict, time_ind)})
            junction_counter += 1 # update junction counter

    pickle.dump(junction_dict_global, open("data/junction_dictionaries/junction_dict.pkl", "wb"))  # save pickled dictionary
    return junction_dict_global

def get_features_from_graph(graph, scaling_dict, time_ind):

    inlet_data = graph.nodes["inlet"].data["inlet_features"].numpy()
    outlet_data = graph.nodes["outlet"].data["outlet_features"].numpy()

    outlet_dP = graph.nodes["outlet"].data["unsteady_outlet_dP"].numpy()
    outlet_flows = graph.nodes["outlet"].data["unsteady_outlet_flows"].numpy()[:, time_ind]

    features = {"outlet_dP" : outlet_dP,
                "inlet_area": np.asarray(np.pi * np.square(inv_scale(scaling_dict, inlet_data[:,0], "inlet_radius"))).reshape(1,-1), # add inlet area (once per junction)
                "outlet_flow": outlet_flows, # add outlet flow (once per junction)
                "inlet_flow": np.asarray(sum(outlet_flows)).reshape(1,-1),
                "outlet_area": np.pi * np.square(inv_scale(scaling_dict, outlet_data[:,0], "outlet_radius").reshape(2,-1)), # add outlet area (once per junction)
                "outlet_angle": inv_scale(scaling_dict, outlet_data[:,1], "angle").reshape(2,-1),
                "inlet_angle": np.asarray(0*outlet_data[:,1]).reshape(1,-1) # add outlet area (once per junction)
                }
    features.update({"inlet_velocity": features["inlet_flow"].reshape(1,-1)/features["inlet_area"].reshape(1,-1)})
    features.update({"outlet_velocity": features["outlet_flow"].reshape(2,-1)/features["outlet_area"].reshape(2,-1)})
    return features

def graphs_to_junction_dict_steady(graph_list, scaling_dict):

    junction_dict_global = {}
    junction_counter = 0 # count total number of junctions

    for graph in graph_list: # loop over all data points
        base_junction = get_features_from_graph_steady(graph, scaling_dict)
        num_time_steps = 100
        num_flows = graph.nodes["outlet"].data["outlet_flows"].numpy().shape[1]
        min_flow1 = graph.nodes["outlet"].data["outlet_flows"].numpy()[0][0]
        max_flow1 = graph.nodes["outlet"].data["outlet_flows"].numpy()[0][num_flows-1]
        flow1_cont = np.linspace(min_flow1, max_flow1, num_time_steps)
        min_flow2 = graph.nodes["outlet"].data["outlet_flows"].numpy()[1][0]
        max_flow2 = graph.nodes["outlet"].data["outlet_flows"].numpy()[1][num_flows-1]
        flow2_cont = np.linspace(min_flow2, max_flow2, num_time_steps)

        for time_ind in range(num_time_steps):
            junction = copy.deepcopy(base_junction)
            junction.update({"outlet_flow": np.asarray([flow1_cont[time_ind], flow2_cont[time_ind]]),
                                    "inlet_flow": np.asarray([flow1_cont[time_ind] + flow2_cont[time_ind]])})
            junction.update({"inlet_velocity": junction["inlet_flow"].reshape(1,-1)/junction["inlet_area"].reshape(1,-1)})
            junction.update({"outlet_velocity": junction["outlet_flow"].reshape(2,-1)/junction["outlet_area"].reshape(2,-1)})
            junction_dict_global.update( # add junction if missing
                    {junction_counter: junction})
            junction_counter += 1 # update junction counter
    pickle.dump(junction_dict_global, open("data/junction_dictionaries/junction_dict_steady.pkl", "wb"))  # save pickled dictionary
    print("Done grouping junctions.")
    return junction_dict_global

def get_features_from_graph_steady(graph, scaling_dict):

    inlet_data = graph.nodes["inlet"].data["inlet_features"].numpy()
    outlet_data = graph.nodes["outlet"].data["outlet_features"].numpy()

    outlet_dP = graph.nodes["outlet"].data["outlet_dP"].numpy()
    outlet_flows = graph.nodes["outlet"].data["outlet_flows"].numpy()

    features = {"inlet_area": np.asarray(np.pi *np.square(inv_scale(scaling_dict, inlet_data[:,0], "inlet_radius"))).reshape(1,-1), # add inlet area (once per junction)
                "inlet_flow": np.asarray(sum(outlet_flows)).reshape(1,-1),
                "outlet_area": np.pi * np.square(inv_scale(scaling_dict, outlet_data[:,0], "outlet_radius")).reshape(2,-1), # add outlet area (once per junction)
                "outlet_angle": inv_scale(scaling_dict, outlet_data[:,1], "angle").reshape(2,-1),
                "inlet_angle": np.asarray(0*outlet_data[:,1]).reshape(1,-1) # add outlet area (once per junction)
                }

    return features
