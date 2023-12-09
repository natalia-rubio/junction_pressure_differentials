import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *
import dgl
import tensorflow as tf

from util.regression.neural_network.graphnet_nn import GraphNet
from util.regression.neural_network.training_nn import *
from dgl.data.utils import load_graphs

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'font.size': 18})



def get_model_name(network_params, train_params, seed, num_geos):

    """
    get unique name for model identifying network and training parameteras, train_dataset
    """

    anatomy = "aorta"
    model_name = f"{network_params['hl_mlp']}_hl_{network_params['latent_size_mlp']}_lsmlp_{str(train_params['learning_rate'])[0:6].replace('.', '_')}_lr_{str(train_params['lr_decay'])[0:5].replace('.', '_')}_lrd_{str(train_params['weight_decay'])[0:5].replace('.', '_')}_wd_bs_{train_params['batch_size']}_nepochs_{train_params['nepochs']}_seed_{seed}_geos_{num_geos}"

    return model_name

def save_model(gnn_model, model_name):

    if not os.path.exists(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/neural_network/steady/{model_name}"):
        os.mkdir(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/neural_network/steady/{model_name}")
    gnn_model.nn_model.save(f"/home/nrubio/Desktop/junction_pressure_differentials/results/models/neural_network/steady/{model_name}")

    return

def train_and_val_gnn(anatomy, seed = 0, num_geos = 10, num_flows = "none", graph_arr = 0, model_list = 0, unsteady = False, config = None):

    train_dataset_loc = f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/train_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset"
    val_dataset_loc =f"/home/nrubio/Desktop/junction_pressure_differentials/data/dgl_datasets/{anatomy}/val_{anatomy}_num_geos_{num_geos}_seed_{seed}_dataset"
    if not unsteady:
        train_dataset_loc+="_steady"
        val_dataset_loc+="_steady"
    train_dataset = load_dict(train_dataset_loc)
    print(train_dataset_loc)
    val_dataset = load_dict(val_dataset_loc)

    network_params = {
                   'latent_size_mlp': 52, # 48,# 20,
                   'out_size': 2,
                   'process_iterations': 1,
                   'hl_mlp':1,
                   'num_inlet_ft' : 2,
                   'num_outlet_ft': 3,
                   'unsteady': unsteady,
                   'output_name': "outlet_coefs"}
    if unsteady:
        network_params["out_size"] = 3

    train_params = {'learning_rate': 0.0931,# 0.018, #0.1,
                    'lr_decay': 0.00834, #0.031, #0.5,
                    'batch_size': 29, #24,# int(np.ceil(len(train_dataset)/10)),
                    'nepochs': 300,
                    'weight_decay': 10**(-5),
                    'optimizer_name' : "adam"}

    if config != None:
        network_params["latent_size_mlp"] = config["latent_size_mlp"]
        network_params["hl_mlp"] = config["hl_mlp"]
        train_params["lr"] = config["lr"]
        train_params["lr_decay"] = config["lr_decay"]
        train_params["batch_size"] = config["batch_size"]

    model_name = get_model_name(network_params = network_params, train_params = train_params, seed = seed, num_geos = num_geos)
    print(f"Launching training.")
    gnn_model = GraphNet(anatomy, network_params, unsteady)
    gnn_model, val_mse, train_mse = train_gnn_model(anatomy,
                                                    gnn_model,
                                                      train_dataset,
                                                      val_dataset,
                                                      train_params = train_params,
                                                      network_params = network_params,
                                                      model_name = model_name)


    save_model(gnn_model, model_name)

    print(f"Train MSE: {train_mse}.  Validation MSE: {val_mse}")

    return train_mse, val_mse, model_name

if __name__ == "__main__":
    train_mse, val_mse, model_name = train_and_val_gnn(anatomy = "Aorta_rand", num_geos = 110,  seed = 0, unsteady = False, config = None)
    #train_mse, val_mse, model_name = train_and_val_gnn(anatomy = "mynard_rand", num_geos = 187,  seed = 0, unsteady = False, config = None)
    print(f"Train MSE: {train_mse}.  Val MSE {val_mse}.")
