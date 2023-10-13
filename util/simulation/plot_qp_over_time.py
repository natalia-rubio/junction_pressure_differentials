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
    geo = "Aorta_5"
    unsteady_result_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{geo}/unsteady_red_sol"
    unsteady_soln_dict = load_dict(unsteady_result_dir)

    posi_Q_ind = np.all(unsteady_soln_dict["flow_in_time"] > 0, axis = 1)

    unsteady_soln_dict["flow_in_time"] = unsteady_soln_dict["flow_in_time"][posi_Q_ind,:]
    unsteady_soln_dict["pressure_in_time"] = unsteady_soln_dict["pressure_in_time"][posi_Q_ind,:]

    unsteady_soln_dict["flow_in_time"] = unsteady_soln_dict["flow_in_time"][:80, :]
    unsteady_soln_dict["pressure_in_time"] = unsteady_soln_dict["pressure_in_time"][:80, :]

    dQdt_unsteady = (unsteady_soln_dict["flow_in_time"][1:,:] - unsteady_soln_dict["flow_in_time"][:-1,:])/0.002
    Q_unsteady = unsteady_soln_dict["flow_in_time"]
    P_unsteady = unsteady_soln_dict["pressure_in_time"]/1333

    plt.clf()
    plt.plot(np.linspace(1, 80, 80), Q_unsteady[:,0], label = 'Inlet', c = "royalblue", linewidth=2)
    plt.plot(np.linspace(1, 80, 80), Q_unsteady[:,1], label = 'Outlet 1', c = "salmon", linewidth=2)
    plt.plot(np.linspace(1, 80, 80), Q_unsteady[:,2], label = 'Outlet 2', c = "seagreen", linewidth=2)
    plt.plot(np.linspace(1, 80, 80), Q_unsteady[:,1]+Q_unsteady[:,2], label = 'Outlet Sum', c = "peru", linewidth=4, linestyle = "--")
    # plt.plot(np.asarray(flow_tensor), np.asarray(tf.reshape(pred_dP_steady[0,:], [-1,]))/1333, label = 'RR (NN)', c = "seagreen", linewidth=2)
    # plt.plot(np.asarray(flow_tensor), dP_mynard, label = 'Unified0D+', c = "salmon", linewidth=2, linestyle ="--")
    # plt.scatter(np.asarray(flow_tensor), np.asarray(dP_tensor)/1333, label = "Simulation", c = "peru", marker = "*", s = 100)

    plt.xlabel("timestep")
    plt.ylabel("$Q \;  (\mathrm{cm^3/s})$")
    #plt.ylabel("$\Delta P$ (mmHg)")
    plt.legend(fontsize="14")
    plt.savefig(f"results/flow_visualization/flow_in_time.pdf", bbox_inches='tight', transparent=True, format = "pdf")

    plt.clf()
    plt.plot(np.linspace(1, 80, 80), P_unsteady[:,0], label = 'Inlet', c = "royalblue", linewidth=2)
    plt.plot(np.linspace(1, 80, 80), P_unsteady[:,1], label = 'Outlet 1', c = "salmon", linewidth=2)
    plt.plot(np.linspace(1, 80, 80), P_unsteady[:,2], label = 'Outlet 2', c = "seagreen", linewidth=2)
    plt.xlabel("timestep")
    plt.ylabel("$P$ (mmHg)")
    plt.legend(fontsize="14")
    plt.savefig(f"results/flow_visualization/pressure_in_time.pdf", bbox_inches='tight', transparent=True, format = "pdf")

    junction_params = load_dict(f"data/synthetic_junctions/{anatomy}/{geo}/junction_params_dict")
    import pdb; pdb.set_trace()
    plt.clf()
    plt.plot(np.linspace(1, 80, 80), P_unsteady[:,0] + (0.5*1.06*(Q_unsteady[:,0]/(np.pi*junction_params["inlet_radius"]**2))**2)/1333, label = 'Inlet', c = "royalblue", linewidth=2)
    plt.plot(np.linspace(1, 80, 80), P_unsteady[:,1] + (0.5*1.06*(Q_unsteady[:,1]/(np.pi*junction_params["outlet1_radius"]**2))**2)/1333, label = 'Outlet 1', c = "salmon", linewidth=2)
    plt.plot(np.linspace(1, 80, 80), P_unsteady[:,2] + (0.5*1.06*(Q_unsteady[:,2]/(np.pi*junction_params["outlet2_radius"]**2))**2)/1333, label = 'Outlet 2', c = "seagreen", linewidth=2)
    # plt.plot(np.linspace(1, 80, 80), P_unsteady[:,2], label = 'Outlet 2', c = "seagreen", linewidth=2)
    # plt.plot(np.asarray(flow_tensor), np.asarray(tf.reshape(pred_dP_steady[0,:], [-1,]))/1333, label = 'RR (NN)', c = "seagreen", linewidth=2)
    # plt.plot(np.asarray(flow_tensor), dP_mynard, label = 'Unified0D+', c = "salmon", linewidth=2, linestyle ="--")
    # plt.scatter(np.asarray(flow_tensor), np.asarray(dP_tensor)/1333, label = "Simulation", c = "peru", marker = "*", s = 100)

    plt.xlabel("timestep")
    plt.ylabel("Total $P$ (mmHg)")
    plt.legend(fontsize="14")
    plt.savefig(f"results/flow_visualization/pressure_dyn_in_time.pdf", bbox_inches='tight', transparent=True, format = "pdf")

    return

plot_unsteady(anatomy = "Aorta_rand")
