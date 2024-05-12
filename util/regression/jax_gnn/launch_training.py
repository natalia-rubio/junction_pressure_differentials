import sys
sys.path.append("/Users/Natalia/Desktop/junction_pressure_differentials")
from util.regression.jax_nn.nn_model import NeuralNet
from util.regression.jax_nn.train_nn import train_nn

def launch_training(anatomy, set_type, network_params, optimizer_params, training_params):
    
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, anatomy, set_type, training_params)
    return

if __name__ == "__main__":
    anatomy = "AP"
    set_type = "random"
    network_params = {"num_input_features": 8,
                      "num_layers": 3,
                      "layer_width": 50,
                      "num_output_features": 3,
                      "anatomy": anatomy,
                      "set_type": set_type}
    
    training_params = {"num_epochs": 1000, 
                       "batch_size": 20}
    
    optimizer_params = {"step_size": 0.002,
                        "init" : 0.01,
                        "transition_steps": 700,
                        "decay_rate" : 0.1}
    
    launch_training(anatomy, set_type, network_params, optimizer_params, training_params)