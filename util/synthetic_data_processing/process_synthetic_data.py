import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *
from util.synthetic_data_processing.extract_synthetic_data import *
from util.synthetic_data_processing.synthesize_synthetic_data import *
from util.synthetic_data_processing.assemble_graphs import *
from util.synthetic_data_processing.train_val_split import *

anatomy = "mynard"
collect_synthetic_results_steady(anatomy)
print("Extracted simulation results.")

get_coefs_steady(anatomy)
print("Fitted dP(Q) coefficients.")

get_geo_scalings_steady(anatomy)
print("Generated scaling dictionary.")

assemble_graphs_steady(anatomy)
print("Assembled graphs.")

generate_train_val_datasets(anatomy)
print("Train and validation datasets ready.")
