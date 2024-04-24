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

tol = 0.05
print("Checking pulmonary junctions.")
pulmo_dict = load_dict("data/characteristic_value_dictionaries/Pulmo_rand_synthetic_data_dict")
for i, inlet_radius in enumerate(pulmo_dict["inlet_radius"]):
    print(inlet_radius)
    if i%2 != 0:
        continue
    if abs((inlet_radius - 0.30)/0.30) <= tol:
        if abs((pulmo_dict["outlet_radius"][i] - 0.20)/0.20) <= tol:
            if abs((pulmo_dict["outlet_radius"][i+1] - 0.20)/0.20) <= tol:
                import pdb; pdb.set_trace()

print("Checking aorta junctions.")
aorta_dict = load_dict("data/characteristic_value_dictionaries/Aorta_rand_synthetic_data_dict")
for i, inlet_radius in enumerate(aorta_dict["inlet_radius"]):
    print(inlet_radius)
            if abs((pulmo_dict["outlet_radius"][i+1] - 0.30)/0.30) <= tol:
                import pdb; pdb.set_trace()
