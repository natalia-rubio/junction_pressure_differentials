import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *
from util.synthetic_data_processing.extract_synthetic_data import *
from util.synthetic_data_processing.synthesize_synthetic_data import *
from util.synthetic_data_processing.assemble_graphs import *
from util.synthetic_data_processing.train_val_split import *

anatomy = sys.argv[1];
collect_synthetic_results(anatomy = anatomy, require4 = False, unsteady = True)
print("Extracted simulation results.")

get_coefs(anatomy = anatomy, rm_low_r2 = True, unsteady = True)
print("Fitted dP(Q) coefficients.")

get_geo_scalings(anatomy, unsteady = True)
print("Generated scaling dictionary.")

assemble_graphs(anatomy, unsteady = True)
print("Assembled graphs.")

generate_train_val_datasets(anatomy)
print("Train and validation datasets ready.")
