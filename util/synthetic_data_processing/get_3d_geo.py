import vtk
import os
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy as v2n
import sys
sys.path.append("/users/Natalia/Desktop/junction_pressure_differentials")
from util.tools.get_bc_integrals import get_res_names
from util.tools.junction_proc import *
from util.tools.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import scipy
import pickle
import jraph
import jax.numpy as jnp
import networkx as nx

def get_3d_geo(anatomy, set_type, geometry):
    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_{set_type}_synthetic_data_dict")
    offset = 10
    try:
        fpath_1dsol = f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geometry}/1dsol_flow_solution_0.vtp"
    except:
        print(f"File not found: {fpath_1dsol}")
    soln = read_geo(fpath_1dsol).GetOutput()  # get 3d flow data

    soln_array = get_all_arrays(soln)
    points = v2n(soln.GetPoints().GetData()) # num_points x 3
    #Extract Geometry ----------------------------------------------------
    direction = soln_array["CenterlineSectionNormal"]  # vector normal direction
    direction_norm = np.linalg.norm( direction, axis=1, keepdims=True)  # norm of direction vector
    direction = np.divide(direction,direction_norm)  # num_points x 3 # normalized direction vector
    area = soln_array["area"]
    pt_id = soln_array["GlobalNodeId"].astype(int)
    num_pts = np.size(pt_id)  # number of points in mesh
    branch_id = soln_array["BranchIdTmp"].astype(int)
    junction_id = soln_array["BifurcationId"].astype(int)

    inlet_pts = np.where(branch_id == 0)[0]
    outlet1_pts = np.where(branch_id == 1)[0]
    outlet2_pts = np.where(branch_id == 2)[0]
    downsample_pts = 10

    inlet_locs = downsample((points[inlet_pts])[np.argsort(pt_id[inlet_pts])], num_samples = downsample_pts)
    outlet1_locs = downsample((points[outlet1_pts])[np.argsort(pt_id[outlet1_pts])], num_samples = downsample_pts)
    outlet2_locs = downsample((points[outlet2_pts])[np.argsort(pt_id[outlet2_pts])], num_samples = downsample_pts)
    locs = np.concatenate((inlet_locs, outlet1_locs, outlet2_locs), axis=0)

    inlet_tan = downsample(direction[inlet_pts][np.argsort(pt_id[inlet_pts])], num_samples = downsample_pts)
    outlet1_tan = downsample(direction[outlet1_pts][np.argsort(pt_id[outlet1_pts])], num_samples = downsample_pts)
    outlet2_tan = downsample(direction[outlet2_pts][np.argsort(pt_id[outlet2_pts])], num_samples = downsample_pts)
    tans = np.concatenate((inlet_tan, outlet1_tan, outlet2_tan), axis=0)

    inlet_area = downsample(area[inlet_pts][np.argsort(pt_id[inlet_pts])], num_samples = downsample_pts)
    outlet1_area = downsample(area[outlet1_pts][np.argsort(pt_id[outlet1_pts])], num_samples = downsample_pts)
    outlet2_area = downsample(area[outlet2_pts][np.argsort(pt_id[outlet2_pts])], num_samples = downsample_pts)
    areas = np.concatenate((inlet_area, outlet1_area, outlet1_area), axis=0)

    # Label node ids for the graph
    
    inlet_node_ids, main_outlet_node_ids, aux_outlet_node_ids, type_array = get_graph_node_ids(inlet_locs, outlet1_locs, outlet2_locs)
    downstream_neighbors, upstream_neighbors = get_neighbors(inlet_node_ids, main_outlet_node_ids, aux_outlet_node_ids)
    senders = downstream_neighbors[1:]-1 # np.concatenate((downstream_neighbors, upstream_neighbors))
    receivers = upstream_neighbors[1:]-1 # np.concatenate((upstream_neighbors, downstream_neighbors))

    # num_nodes x num_features
    node_features = np.concatenate((
        areas[upstream_neighbors].reshape(-1,1),
        np.linalg.norm(locs[upstream_neighbors] - locs[downstream_neighbors], axis = 1).reshape(-1,1),
        np.linalg.norm(tans[upstream_neighbors] - tans[downstream_neighbors], axis = 1).reshape(-1,1),
        type_array[upstream_neighbors,:]), axis=1)
    
    edge_features = np.concatenate((np.linalg.norm(locs[senders] - locs[receivers], axis = 1).reshape(-1,1),
                                    np.linalg.norm(tans[senders] - tans[receivers], axis = 1).reshape(-1,1)), axis=1)
    
    
    globals = np.asarray([char_val_dict["coef_a"][0], char_val_dict["coef_b"][0], 0] + \
                         char_val_dict["flow_list"][0] + char_val_dict["dP_list"][0])

    graph = jraph.GraphsTuple(nodes = jnp.array(node_features),
                              edges = jnp.array(edge_features), 
                              receivers = jnp.array(receivers),
                              senders = jnp.array(senders),
                              globals = globals,
                              n_node = node_features.shape[0],
                              n_edge = edge_features.shape[0])
    return graph

def get_graph_node_ids(inlet_locs, outlet1_locs, outlet2_locs):
    inlet_node_ids = np.linspace(0, 
                                  len(inlet_locs), 
                                  len(inlet_locs), endpoint = False).astype(int)
    main_outlet_node_ids = np.linspace(len(inlet_locs), 
                                        len(inlet_locs) + len(outlet1_locs), 
                                        len(outlet1_locs), endpoint = False).astype(int)
    aux_outlet_node_ids = np.linspace(len(inlet_locs) + len(outlet1_locs), 
                                       len(inlet_locs) + len(outlet1_locs) + len(outlet2_locs), 
                                       len(outlet2_locs), endpoint = False).astype(int)
    
    type_array = np.zeros((len(inlet_node_ids) + len(main_outlet_node_ids) + len(aux_outlet_node_ids), 3))
    type_array[inlet_node_ids, 0] = 1
    type_array[main_outlet_node_ids, 1] = 1
    type_array[aux_outlet_node_ids, 2] = 1

    return inlet_node_ids, main_outlet_node_ids, aux_outlet_node_ids, type_array

def get_neighbors(inlet_node_ids, main_outlet_node_ids, aux_outlet_node_ids):
    downstream_neighbors = np.concatenate((inlet_node_ids,
                                           main_outlet_node_ids[0:-1],
                                           np.asarray(inlet_node_ids[-1]).reshape(-1,),
                                           aux_outlet_node_ids[:-1]), axis=0)
    
    upstream_neighbors = np.concatenate((inlet_node_ids[1:], 
                                         main_outlet_node_ids, 
                                         aux_outlet_node_ids), axis = 0)
    return downstream_neighbors, upstream_neighbors

def downsample(input_array, num_samples):
    x_locs = np.linspace(0, len(input_array), len(input_array))
    x_new = np.linspace(0, len(input_array), num_samples)
    interp_fun = scipy.interpolate.interp1d(x_locs, input_array, axis = 0)
    return interp_fun(x_new)

def convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
  nodes, edges, receivers, senders, _, _, _ = jraph_graph
  nx_graph = nx.DiGraph()
  if nodes is None:
    for n in range(jraph_graph.n_node):
      nx_graph.add_node(n)
  else:
    for n in range(jraph_graph.n_node):
      nx_graph.add_node(n, node_feature=nodes[n])
  if edges is None:
    for e in range(jraph_graph.n_edge):
      nx_graph.add_edge(int(senders[e]), int(receivers[e]))
  else:
    for e in range(jraph_graph.n_edge):
      nx_graph.add_edge(
          int(senders[e]), int(receivers[e]), edge_feature=edges[e])
  return nx_graph


def draw_jraph_graph_structure(jraph_graph: jraph.GraphsTuple) -> None:
  nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
  pos = nx.spring_layout(nx_graph)
  nx.draw(nx_graph, pos=pos, with_labels=True, node_size=500, font_color='yellow')
  plt.show()
     


     

if __name__ == "__main__":
    anatomy = "AP"
    set_type = "random"
    geometry = "AP_000"
    graph = get_3d_geo(anatomy, set_type, geometry)
    save_dict(graph, f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geometry}/graph")

    #draw_jraph_graph_structure(graph)
