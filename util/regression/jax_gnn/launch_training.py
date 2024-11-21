import sys
sys.path.append("/Users/Natalia/Desktop/junction_pressure_differentials")
from util.regression.jax_gnn.gnn_model import GraphNeuralNet
from util.regression.jax_gnn.train_gnn import train_gnn

def launch_training(anatomy, set_type, network_params, optimizer_params, training_params):
    
    model = GraphNeuralNet(network_params, optimizer_params)
    train_gnn(model, anatomy, set_type, training_params)
    return

if __name__ == "__main__":
    anatomy = "Aorta"
    set_type = "random"
    network_params = {"num_input_features": 6,
                    "num_encoder_layers": 1,
                    "num_passing_layers": 1,
                    "num_decoder_layers": 1,
                    "layer_width": 2,
                    "num_output_features": 3,
                    "num_message_passing_steps": 1,
                    "anatomy": anatomy,
                    "set_type": set_type}
    
    training_params = {"num_epochs": 1000, 
                       "batch_size": 20}
    
    optimizer_params = {"init" : 0.002,
                        "transition_steps": 10000000,
                        "decay_rate" : 0.05}
    
    launch_training(anatomy, set_type, network_params, optimizer_params, training_params)