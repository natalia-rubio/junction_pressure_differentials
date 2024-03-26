import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *
import dgl
import tensorflow as tf
from dgl.data import DGLDataset
from os.path import exists


class DGL_Dataset(DGLDataset):

    def __init__(self, graphs):
        self.graphs = graphs
        super().__init__(name='dgl_dataset')

    def process(self):

        pass

    def __getitem__(self, i):

        return self.graphs[i]

    def __len__(self):

        return len(self.graphs)


def generate_train_val_datasets(anatomy, set_type, dataset_params = {}, unsteady = True):
    seed = 0
    print(unsteady)
    if unsteady:
        graph_list = dgl.load_graphs(f"data/graph_lists/{anatomy}/{set_type}/graph_list")[0]; graph_arr = np.array(graph_list, dtype = "object")
    else:
        graph_list = dgl.load_graphs(f"data/graph_lists/{anatomy}/{set_type}/graph_list_steady")[0]; graph_arr = np.array(graph_list, dtype = "object")
    geo_list = [graph.nodes["inlet"].data["geo_name"][0] for graph in graph_list]
    geo_arr = np.asarray(geo_list)
    num_geos = len(geo_list); print(f"Number of geometries: {num_geos}")
    train_ind, val_ind = get_random_ind(num_pts = len(geo_list), percent_train = 80, seed = seed)

    train_geos = geo_arr[train_ind]; val_geos = geo_arr[val_ind]
    train_graph_list =  graph_arr[train_ind]
    val_graph_list =  graph_arr[val_ind]
    print(f"train geos: {train_geos} \nval geos: {val_geos}")
    print(train_graph_list[0].nodes["inlet"])
    assert len(list(set(train_graph_list).intersection(val_graph_list))) == 0, "Common geometries between train and val sets."

    train_dataset = DGL_Dataset(list(graph_arr[train_ind]))
    val_dataset = DGL_Dataset(list(graph_arr[val_ind]))

    if not os.path.exists(f"data/split_indices"):
        os.mkdir(f"data/split_indices")
    if not os.path.exists(f"data/split_indices/{anatomy}"):
        os.mkdir(f"data/split_indices/{anatomy}")
    if not os.path.exists(f"data/split_indices/{anatomy}/{set_type}"):
        os.mkdir(f"data/split_indices/{anatomy}/{set_type}")
    np.save(f"data/split_indices/{anatomy}/{set_type}/train_ind_{anatomy}_num_geos_{num_geos}_seed_{seed}", train_ind)
    np.save(f"data/split_indices/{anatomy}/{set_type}/val_ind_{anatomy}_num_geos_{num_geos}_seed_{seed}", val_ind)

    if not os.path.exists(f"data/graph_lists/{anatomy}"):
        os.mkdir(f"data/graph_lists/{anatomy}")
    if not os.path.exists(f"data/graph_lists/{anatomy}/{set_type}"):
        os.mkdir(f"data/graph_lists/{anatomy}/{set_type}")
    if unsteady:
        save_dict(train_graph_list, f"data/graph_lists/{anatomy}/{set_type}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_graph_list")
        save_dict(val_graph_list, f"data/graph_lists/{anatomy}/{set_type}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_graph_list")
    else:
        save_dict(train_graph_list, f"data/graph_lists/{anatomy}/{set_type}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_graph_list_steady")
        save_dict(val_graph_list, f"data/graph_lists/{anatomy}/{set_type}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_graph_list_steady")

    if not os.path.exists(f"data/dgl_datasets"):
        os.mkdir(f"data/dgl_datasets")
    if not os.path.exists(f"data/dgl_datasets/{anatomy}"):
        os.mkdir(f"data/dgl_datasets/{anatomy}")
    if not os.path.exists(f"data/dgl_datasets/{anatomy}/{set_type}"):
        os.mkdir(f"data/dgl_datasets/{anatomy}/{set_type}")
    if unsteady:
        save_dict(train_dataset, f"data/dgl_datasets/{anatomy}/{set_type}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset")
        save_dict(val_dataset, f"data/dgl_datasets/{anatomy}/{set_type}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset")
    else:
        save_dict(train_dataset, f"data/dgl_datasets/{anatomy}/{set_type}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset_steady")
        save_dict(val_dataset, f"data/dgl_datasets/{anatomy}/{set_type}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset_steady")
    print(f"total geos: {num_geos}")
    return train_dataset, val_dataset
