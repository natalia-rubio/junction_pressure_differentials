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

    pulmo_graph_list_name = f"data/graph_lists/Pulmo_rand/val_Pulmo_rand_num_geos_130_seed_0_graph_list"

    if anatomy == "Pulmo":
        #model_name = pulmo_model_name
        graph_list_name = pulmo_graph_list_name
        geo_list = [1]#[25, 9, 7, 4, 24] #[16,9,8,1,24]#
        rmse_list = [0.86, 2.5, 3.6, 4.6, 7.6]#[0.29, 0.51, 0.70, 0.88, 2.77]
    elif anatomy == "Aorta":
        model_name = aorta_model_name
        graph_list_name = aorta_graph_list_name
        geo_list = [13, 9, 15, 3, 12] #[2, 20, 12, 9, 15]
        rmse_list = [1.9, 4.0, 5.4, 7.6, 10] #[0.90, 1.1, 1.3, 1.6, 7.4]

    print(graph_list_name)
    graph_list = load_dict(graph_list_name)
    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_rand_scaling_dict")


    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize = (4, 4), sharex = True, sharey = True, layout = "constrained")
    #fig.tight_layout()
    #fig.subplots_adjust(wspace=0.2,hspace=0.2)
    for i in range(len(geo_list)):
        ax = axs
        graph = graph_list[i]
        master_tensor = get_master_tensors_unsteady([graph])

        input_tensor = master_tensor[0]
        output_tensor = master_tensor[1][0,:]
        flow_tensor = master_tensor[2][0,:]
        flow_der_tensor = master_tensor[3][0,:]
        dP_tensor = master_tensor[4][0,:]

        #ax.scatter(np.asarray(flow_tensor)[0:40], np.asarray(dP_tensor)[0:40]/1333, label = "Simulation", c = "seagreen", marker = "*", s = 70)
        ax.scatter(np.asarray(flow_tensor)[0:], np.asarray(dP_tensor)[0:]/1333, label = "Simulation", c = "salmon", marker = "*", s = 70)

    #fig.subplots_adjust(hspace=0.1)
    fig.text(0.5, -0.02, "$Q \;  (\mathrm{cm^3/s})$", ha = "center")
    fig.text(-0.08, 0.5, "$\Delta P$ (mmHg)", rotation = "vertical", va = "center")
    fig.savefig(f"results/model_visualization/{anatomy}_unsteady_flow_vs_predicted_dps_single.pdf", bbox_inches='tight', transparent=True, format = "pdf")

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
