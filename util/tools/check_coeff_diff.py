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

def check_coef_dif(char_val_dict):
    steady_a = np.asarray(char_val_dict["coef_a"])
    steady_b = np.asarray(char_val_dict["coef_b"])

    unsteady_a = np.asarray(char_val_dict["coef_a_UO"])
    unsteady_b = np.asarray(char_val_dict["coef_b_UO"])

    print(f"A RMSE Percent Difference: {np.sqrt(np.linalg.norm((steady_a - unsteady_a)/steady_a))}")
    print(f"B RMSE Percent Difference: {np.sqrt(np.linalg.norm((steady_b - unsteady_b)/steady_b))}")
    #import pdb; pdb.set_trace()
    return

print("Aorta Dataset:")
aorta_dict = load_dict("data/characteristic_value_dictionaries/Aorta_rand_synthetic_data_dict")
check_coef_dif(aorta_dict)

print("Pulmo Dataset:")
pulmo_dict = load_dict("data/characteristic_value_dictionaries/Pulmo_rand_synthetic_data_dict")
check_coef_dif(pulmo_dict)
