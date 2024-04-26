import sys
sys.path.append("/Users/Natalia/Desktop/junction_pressure_differentials")
from util.regression.jax_nn.nn_model import NeuralNet
from util.regression.jax_nn.train_nn import train_nn

def launch_training(anatomy, set_type, network_params, training_params):
    
    model = NeuralNet(network_params)
    train_nn(model, anatomy, set_type, training_params)
    return

if __name__ == "__main__":
    anatomy = "AP"
    set_type = "random"
    network_params = {"num_input_features": 8,
                      "num_layers": 3,
                      "layer_width": 10,
                      "num_output_features": 3,
                      "anatomy": anatomy,
                      "set_type": set_type}
    
    training_params = {"num_epochs": 10, 
                       "batch_size": 20,
                       "step_size": 0.0001}
    
    launch_training(anatomy, set_type, network_params, training_params)