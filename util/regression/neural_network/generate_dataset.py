import dgl
import tensorflow as tf
import numpy as np
from dgl.data import DGLDataset
from os.path import exists
import pickle
import pdb
import math
import random
import os

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_random_ind(num_pts, percent_train = 85 , seed = 0):

    ind = np.linspace(0,num_pts, num_pts, endpoint = False).astype(int); rng = np.random.default_rng(seed)
    train_ind = rng.choice(ind, size = int(num_pts * 0.01 * percent_train), replace = False)
    #print(train_ind)
    val_ind = ind[np.isin(ind, train_ind, invert = True)]
    return train_ind, val_ind

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

def generate_dataset_synthetic_vary_flows(dataset_params = {}, seed = 0, num_flows = 2, graph_arr = 0, model_list = 0):
    dataset_name = f'{dataset_params["source"]}_{dataset_params["features"]}'
    # print(f"Generating Bifurcation Dataset ({dataset_name})")
    # graph_list = dgl.load_graphs(f"/home/nrubio/Desktop/junction_GNN/data/master_data/graph_list_{dataset_name}")[0]; graph_arr = np.array(graph_list, dtype = "object")
    # model_arr = np.load(f"/home/nrubio/Desktop/junction_GNN/data/master_data/model_list_{dataset_name}.npy");
    # model_list = [model[5:] for model in list(model_arr)];
    model_arr = np.array(model_list)

    random.seed(seed)
    geo_list = list(dict.fromkeys(model_list)); random.shuffle(geo_list); geo_arr = np.array(geo_list)
    num_geos = len(geo_list)
    print(f"Total number of unique models: {len(geo_list)}.")

    train_graph_arr = np.empty((0,))
    val_graph_arr = np.empty((0,))

    train_graph_arr = np.empty((0,))
    for geo in geo_list:
        geo_graphs = graph_arr[np.where(model_arr == geo)]
        np.random.shuffle(geo_graphs)
        num_train = min(geo_graphs.size-1, num_flows)
        train_graph_arr = np.append(train_graph_arr, geo_graphs[0:num_train])
        val_graph_arr = np.append(val_graph_arr, geo_graphs[num_train:min(int(num_train + num_flows/2), geo_graphs.size)])

    train_graph_list = list(train_graph_arr)
    val_graph_list = list(val_graph_arr)
    assert len(list(set(train_graph_list).intersection(val_graph_list))) == 0, "Common geometries between train and val sets."

    train_dataset = DGL_Dataset(train_graph_list)
    val_dataset = DGL_Dataset(val_graph_list)

    save_dict(train_graph_list, f"/home/nrubio/Desktop/junction_GNN/data/graph_lists/train_{dataset_name}_num_geos_{num_geos}_num_flows_{num_flows}_seed_{seed}_graph_list")
    save_dict(val_graph_list, f"data/graph_lists/val_{dataset_name}_num_geos_{num_geos}_num_flows_{num_flows}_seed_{seed}_graph_list")

    save_dict(train_dataset, f"/home/nrubio/Desktop/junction_GNN/data/datasets/train_{dataset_name}_num_geos_{num_geos}_num_flows_{num_flows}_seed_{seed}_dataset")
    save_dict(val_dataset, f"/home/nrubio/Desktop/junction_GNN/data/datasets/val_{dataset_name}_num_geos_{num_geos}_num_flows_{num_flows}_seed_{seed}_dataset")
    return train_dataset, val_dataset

def generate_dataset_synthetic(dataset_name, dataset_params = {}, seed = 0, num_geos = 130, graph_arr = 0, model_list = 0):

    print(f"Generating Bifurcation Dataset ({dataset_name})")
    # graph_list = dgl.load_graphs(f"data/master_data/graph_list_{dataset_name}")[0]; graph_arr = np.array(graph_list, dtype = "object")
    # model_arr = np.load(f"data/master_data/model_list_{dataset_name}.npy");
    # model_list = [model[5:] for model in list(model_arr)];


    if exists(f"data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset"):
        train_dataset = load_dict(f"data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
        val_dataset = load_dict(f"data/datasets/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    else:
        print(f"Generating Bifurcation Dataset ({dataset_name})")
        if graph_arr == 0:
            graph_list = dgl.load_graphs(f"data/master_data/graph_list_{dataset_name}")[0]; graph_arr = np.array(graph_list, dtype = "object")
        else:
            graph_list = list(graph_arr)

        if model_list == 0:
            model_arr = np.load(f"data/master_data/model_list_{dataset_name}.npy");
        else:
            model_arr = np.array(model_list)

        if dataset_params["source"] == "synthetic" or dataset_params["source"] == "synthetic_steady":
            model_list = [model[5:] for model in list(model_arr)]; model_arr = np.array(model_list)
        elif dataset_params["source"] == "vmr":
            model_list = [model[0:9] for model in list(model_arr)]; model_arr = np.array(model_list)

        random.seed(seed)
        geo_list = list(dict.fromkeys(model_list)); random.shuffle(geo_list); geo_arr = np.array(geo_list)
        print(f"Total number of unique models: {len(geo_list)}.")
        print(f"Total number of graphs: {len(graph_list)}.")


    random.seed(seed)
    geo_list = list(dict.fromkeys(model_list)); random.shuffle(geo_list); geo_arr = np.array(geo_list)
    print(f"Total number of unique models: {len(geo_list)}.")
    #import pdb; pdb.set_trace()

    train_ind, val_ind = get_random_ind(num_pts = len(geo_list), seed = seed)
    num_train = int(0.85*num_geos); num_val = num_geos - num_train

    train_geos = geo_arr[train_ind[:num_train]]; val_geos = geo_arr[val_ind[:num_val]]
    print(f"train geos: {train_geos} \nval geos: {val_geos}")
    train_graph_list = (graph_arr[np.isin(model_arr, train_geos)])
    val_graph_list = (graph_arr[np.isin(model_arr, val_geos)])
    assert len(list(set(train_graph_list).intersection(val_graph_list))) == 0, "Common geometries between train and val sets."

    train_dataset = DGL_Dataset(list(graph_arr[np.isin(model_arr, train_geos)]))
    val_dataset = DGL_Dataset(list(graph_arr[np.isin(model_arr, val_geos)]))

    np.save(f"/home/nrubio/Desktop/junction_GNN/data/split_indices/train_ind_{dataset_name}_num_geos_{num_geos}_seed_{seed}", train_ind)
    np.save(f"/home/nrubio/Desktop/junction_GNN/data/split_indices/val_ind_{dataset_name}_num_geos_{num_geos}_seed_{seed}", val_ind)

    save_dict(train_graph_list, f"/home/nrubio/Desktop/junction_GNN/data/graph_lists/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_graph_list")
    save_dict(val_graph_list, f"/home/nrubio/Desktop/junction_GNN/data/graph_lists/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_graph_list")

    save_dict(train_dataset, f"/home/nrubio/Desktop/junction_GNN/data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    save_dict(val_dataset, f"/home/nrubio/Desktop/junction_GNN/data/datasets/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    return train_dataset, val_dataset

def generate_dataset_vmr(dataset_name, dataset_params = {}, seed = 0, num_geos = 130, graph_arr = 0, model_list = 0):
    print(f"Generating Dataset ({dataset_name})")

    if exists(f"data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset"):
        train_dataset = load_dict(f"data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
        val_dataset = load_dict(f"data/datasets/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    else:
        print(f"Generating Bifurcation Dataset ({dataset_name})")
        if graph_arr == 0:
            graph_list = dgl.load_graphs(f"data/master_data/graph_list_{dataset_name}")[0]; graph_arr = np.array(graph_list, dtype = "object")
        else:
            graph_list = list(graph_arr)

        if model_list == 0:
            model_arr = np.load(f"data/master_data/model_list_{dataset_name}.npy");
        else:
            model_arr = np.array(model_list)

        if dataset_params["source"] == "synthetic":
            model_list = [model[5:] for model in list(model_arr)]; model_arr = np.array(model_list)
        elif dataset_params["source"] == "vmr":
            model_list = [model[0:11] for model in list(model_arr)]; model_arr = np.array(model_list)

        random.seed(seed)
        geo_list = list(dict.fromkeys(model_list)); random.shuffle(geo_list); geo_arr = np.array(geo_list)
        print(f"Total number of unique models: {len(geo_list)}.")
        print(f"Total number of graphs: {len(graph_list)}.")

    train_ind, val_ind = get_random_ind(num_pts = len(geo_list), seed = seed)
    num_train = int(0.85*num_geos); num_val = num_geos - num_train

    train_geos = geo_arr[train_ind[:num_train]]; val_geos = geo_arr[val_ind[:num_val]]
    print(f"train geos: {train_geos} \nval geos: {val_geos}")
    train_graph_list = (graph_arr[np.isin(model_arr, train_geos)])
    val_graph_list = (graph_arr[np.isin(model_arr, val_geos)])
    assert len(list(set(train_graph_list).intersection(val_graph_list))) == 0, "Common geometries between train and val sets."

    train_dataset = DGL_Dataset(list(graph_arr[np.isin(model_arr, train_geos)]))
    val_dataset = DGL_Dataset(list(graph_arr[np.isin(model_arr, val_geos)]))

    np.save(f"/home/nrubio/Desktop/junction_GNN/data/split_indices/train_ind_{dataset_name}_num_geos_{num_geos}_seed_{seed}", train_ind)
    np.save(f"/home/nrubio/Desktop/junction_GNN/data/split_indices/val_ind_{dataset_name}_num_geos_{num_geos}_seed_{seed}", val_ind)

    save_dict(train_graph_list, f"/home/nrubio/Desktop/junction_GNN/data/graph_lists/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_graph_list")
    save_dict(val_graph_list, f"/home/nrubio/Desktop/junction_GNN/data/graph_lists/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_graph_list")

    save_dict(train_dataset, f"/home/nrubio/Desktop/junction_GNN/data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    save_dict(val_dataset, f"/home/nrubio/Desktop/junction_GNN/data/datasets/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    return train_dataset, val_dataset

def generate_dataset(dataset_params = {}, seed = 0, num_geos = 130, vary_flow = False, num_flows = 10, graph_arr = 0, model_list = 0):

    print(os.getcwd())
    dataset_name = f"{dataset_params['source']}_{dataset_params['features']}_{dataset_params['output']}_{dataset_params['filter']}_{dataset_params['scale_type']}_scale"

    if dataset_params["augmentation"] == "angle":
        dataset_name += "_angle_augment"

    if vary_flow == True:
        print("Varying number of flows")
        print(f"/home/nrubio/Desktop/junction_GNN/data/datasets/train_{dataset_name}_num_geos_{num_geos}_num_flows_{num_flows}_seed_{seed}_dataset")
        if exists(f"/home/nrubio/Desktop/junction_GNN/data/datasets/train_{dataset_name}_num_geos_{num_geos}_num_flows_{num_flows}_seed_{seed}_dataset"):
            train_dataset = load_dict(f"/home/nrubio/Desktop/junction_GNN/data/datasets/train_{dataset_name}_num_geos_{num_geos}_num_flows_{num_flows}_seed_{seed}_dataset")
            val_dataset = load_dict(f"/home/nrubio/Desktop/junction_GNN/data/datasets/val_{dataset_name}_num_geos_{num_geos}_num_flows_{num_flows}_seed_{seed}_dataset")
        else:
            print(f"Generating Bifurcation Dataset ({dataset_name})")
            if dataset_params["source"] == "synthetic":
                train_dataset, val_dataset = generate_dataset_synthetic_vary_flows(dataset_params, seed, num_flows, graph_arr = graph_arr, model_list = model_list)


    else:
        if exists(f"data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset"):
            train_dataset = load_dict(f"/home/nrubio/Desktop/junction_GNN/data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
            val_dataset = load_dict(f"/home/nrubio/Desktop/junction_GNN/data/datasets/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")

        else:

            print(f"Generating Bifurcation Dataset ({dataset_name})")
            if dataset_params["source"] == "synthetic" or dataset_params["source"] == "synthetic_steady":
                train_dataset, val_dataset = generate_dataset_synthetic(dataset_name, dataset_params, seed, num_geos, graph_arr, model_list)

            elif dataset_params["source"] == "vmr":
                train_dataset, val_dataset = generate_dataset_vmr(dataset_name, dataset_params, seed, num_geos)
    return train_dataset, val_dataset

def generate_dataset_aorta(dataset_params = {}):
    seed = 2
    graph_list = dgl.load_graphs(f"/home/nrubio/Desktop/aorta_graph_list")[0]; graph_arr = np.array(graph_list, dtype = "object")
    geo_list = [graph.nodes["inlet"].data["geo_name"][0] for graph in graph_list]
    geo_arr = np.asarray(geo_list)
    num_geos = len(geo_list); print(f"Number of geometries: {num_geos}")
    train_ind, val_ind = get_random_ind(num_pts = len(geo_list), percent_train = 80, seed = seed)
    #val_ind = np.array([0])
    #num_train = int(0.95*num_geos); num_val = num_geos - num_train

    train_geos = geo_arr[train_ind]; val_geos = geo_arr[val_ind]
    train_graph_list =  graph_arr[train_ind]
    val_graph_list =  graph_arr[val_ind]
    print(f"train geos: {train_geos} \nval geos: {val_geos}")

    #assert len(list(set(train_graph_list).intersection(val_graph_list))) == 0, "Common geometries between train and val sets."

    train_dataset = DGL_Dataset(list(graph_arr[train_ind]))
    val_dataset = DGL_Dataset(list(graph_arr[val_ind]))

    dataset_name = "aorta"
    np.save(f"/home/nrubio/Desktop/junction_GNN_aorta/data/split_indices/train_ind_{dataset_name}_num_geos_{num_geos}_seed_{seed}", train_ind)
    np.save(f"/home/nrubio/Desktop/junction_GNN_aorta/data/split_indices/val_ind_{dataset_name}_num_geos_{num_geos}_seed_{seed}", val_ind)

    save_dict(train_graph_list, f"/home/nrubio/Desktop/junction_GNN_aorta/data/graph_lists/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_graph_list")
    save_dict(val_graph_list, f"/home/nrubio/Desktop/junction_GNN_aorta/data/graph_lists/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_graph_list")

    save_dict(train_dataset, f"/home/nrubio/Desktop/junction_GNN_aorta/data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    save_dict(val_dataset, f"/home/nrubio/Desktop/junction_GNN_aorta/data/datasets/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    print(f"total geos: {num_geos}")
    return train_dataset, val_dataset

if __name__ == '__main__':
    generate_dataset_aorta()
    # dataset_params = {"source": "synthetic", # where to extraxt graphs from (synthetic or vmr)
    #                 "features": "full", # complexity of features (full or reduced)
    #                 "junction_type": "y"
    #                 }
    #generate_dataset(dataset_params = dataset_params, seed = 0, num_geos = 240)
    #generate_dataset_synthetic_vary_flows(dataset_params, seed = 0, num_flows = 20)
