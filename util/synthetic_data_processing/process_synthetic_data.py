import sys
sys.path.append("/Users/Natalia/Desktop/junction_pressure_differentials")
from util.tools.basic import *
from util.synthetic_data_processing.extract_synthetic_data import *
from util.synthetic_data_processing.synthesize_synthetic_data import *
# from util.synthetic_data_processing.assemble_graphs import *
# from util.synthetic_data_processing.train_val_split import *

anatomy = sys.argv[1];
set_type= sys.argv[2];
unsteady_text = sys.argv[3] #alse
unsteady = False
if unsteady_text == "unsteady":
    unsteady = True
print(f"Unsteady: {unsteady}")
collect_synthetic_results(anatomy = anatomy, set_type = set_type, require4 =False, unsteady = unsteady)
print("Extracted simulation results.")

get_coefs(anatomy = anatomy, set_type = set_type, rm_low_r2 = True, unsteady = unsteady)
print("Fitted dP(Q) coefficients.")

get_geo_scalings(anatomy, set_type = set_type, unsteady = unsteady)
print("Generated scaling dictionary.")

# assemble_graphs(anatomy, set_type = set_type, unsteady = unsteady)
# print("Assembled graphs.")

# generate_train_val_datasets(anatomy, set_type = set_type, unsteady = unsteady)
# print("Train and validation datasets ready.")
