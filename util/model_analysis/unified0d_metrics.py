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



def plot_unsteady(graph_list):

    scaling_dict = load_dict("data/scaling_dictionaries/Aorta_rand_scaling_dict")
    dP_mynard = []
    dP_true = []
    for j in range(graph_list.size):

        graph = graph_list[j]
        master_tensor = get_master_tensors_steady([graph])
        #
        flow_tensor = master_tensor[2]
        dP_tensor = master_tensor[4]

        # Apply Unified0D Method
        junction_dict_global = graphs_to_junction_dict_steady(graph, scaling_dict)

        flow_arr = flow_tensor.numpy()


        for outlet_ind in range(2):
            for i in range(1,flow_arr.shape[1]):
                # print(f"i = {i}")
                # print(junction_dict_global)
                # print(flow_arr)
                dP_mynard = dP_mynard + [(apply_unified0D_plus(junction_dict_global[i])[outlet_ind] \
                                - dP_poiseuille(flow = flow_arr[outlet_ind, i], radius = junction_dict_global[i]["inlet_radius"][0], length = junction_dict_global[i]["inlet_length"][0]) \
                                - dP_poiseuille(flow = flow_arr[outlet_ind, i], radius = junction_dict_global[i]["outlet_radius"][outlet_ind], length = junction_dict_global[i]["outlet_length"][outlet_ind]))/1333]

                dP_true.append(dP_tensor.numpy()[outlet_ind,i]/1333)

    print(f"RMSE: {rmse_numpy(np.asarray(dP_true).reshape(-1,), np.asarray(dP_mynard).reshape(-1,))}")
    return


anatomy = "Aorta_rand"
num_geos = 110
print("Validation")
graph_list = load_dict(f"data/graph_lists/{anatomy}/val_{anatomy}_num_geos_{num_geos}_seed_0_graph_list")
plot_unsteady(graph_list)
