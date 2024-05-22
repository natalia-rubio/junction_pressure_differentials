import sys
sys.path.append("/Users/natalia/Desktop/junction_pressure_differentials")
from util.tools.basic import *
import jax.numpy as jnp

def get_numpy_arrays(anatomy, set_type, unsteady = False):
    print("Colleting numpy arrays.")
    geo_list = []
    flow_list = []
    dP_list = []
    coef_list = []
    geo_name_list = []

    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_{set_type}_synthetic_data_dict")
    # if unsteady:
    #     scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_scaling_dict")
    # else:
    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")

    if not os.path.exists(f"data/graph_lists/{anatomy}"):
        os.mkdir(f"data/graph_lists/{anatomy}")
    if not os.path.exists(f"data/graph_lists/{anatomy}/{set_type}"):
        os.mkdir(f"data/graph_lists/{anatomy}/{set_type}")

    for i in range(int(len(char_val_dict["inlet_radius"])/2)):

        geo_list.append([
                # scale(scaling_dict, char_val_dict["coef_a"][2*i], "coef_a"),
                # scale(scaling_dict, char_val_dict["coef_b"][2*i], "coef_b"),
                scale(scaling_dict, np.asarray(char_val_dict["inlet_radius"][2*i]), "inlet_radius"),\
                scale(scaling_dict, np.asarray(char_val_dict["inlet_length"][2*i]), "inlet_length"),\

                scale(scaling_dict, np.asarray(char_val_dict["outlet_radius"][2*i]), "outlet_radius"),\
                scale(scaling_dict, np.asarray(char_val_dict["outlet_length"][2*i]), "outlet_length"),\
                scale(scaling_dict, np.asarray(char_val_dict["angle"][2*i]), "angle"),\

                scale(scaling_dict, np.asarray(char_val_dict["outlet_radius"][2*i + 1]), "outlet_radius"),\
                scale(scaling_dict, np.asarray(char_val_dict["outlet_length"][2*i + 1]), "outlet_length"),\
                scale(scaling_dict, np.asarray(char_val_dict["angle"][2*i + 1]), "angle")])

        flow_list.append(char_val_dict["flow_list"][2*i]) # + char_val_dict["flow_list"][2*i + 1])
       
        dP_list.append(char_val_dict["dP_list"][2*i]) # + char_val_dict["dP_list"][2*i + 1])

        geo_name_list.append(char_val_dict["name"][2*i])

        if unsteady:

            unsteady_outlet_flows = np.stack((np.asarray(char_val_dict["unsteady_flow_list"][2*i]).T,
                                np.asarray(char_val_dict["unsteady_flow_list"][2*i + 1]).T))

            unsteady_outlet_flow_ders = np.stack((np.asarray(char_val_dict["unsteady_flow_der_list"][2*i]).T,
                                np.asarray(char_val_dict["unsteady_flow_der_list"][2*i + 1]).T))

            unsteady_outlet_dPs = np.stack((np.asarray(char_val_dict["unsteady_dP_list"][2*i]).T,
                                np.asarray(char_val_dict["unsteady_dP_list"][2*i + 1]).T))

            outlet_coefs = np.asarray([scale(scaling_dict, char_val_dict["coef_a"][2*i: 2*(i+1)], "coef_a"),
                                    scale(scaling_dict, char_val_dict["coef_b"][2*i: 2*(i+1)], "coef_b"),
                                    scale(scaling_dict, char_val_dict["coef_L"][2*i: 2*(i+1)], "coef_L")]).T

            outlet_coefs_UO = np.asarray([scale(scaling_dict, char_val_dict["coef_a_UO"][2*i: 2*(i+1)], "coef_a_UO"),
                                    scale(scaling_dict, char_val_dict["coef_b_UO"][2*i: 2*(i+1)], "coef_b_UO"),
                                    scale(scaling_dict, char_val_dict["coef_L_UO"][2*i: 2*(i+1)], "coef_L_UO")]).T

        else:
            coef_list.append([scale(scaling_dict, char_val_dict["coef_a"][2*i], "coef_a"),
                scale(scaling_dict, char_val_dict["coef_b"][2*i], "coef_b"),
                scale(scaling_dict, char_val_dict["coef_L"][2*i], "coef_L")])
    data_dict = {"geo": jnp.asarray(geo_list), 
                 "flow": jnp.asarray(flow_list), 
                 "dP": jnp.asarray(dP_list), 
                 "coef": jnp.asarray(coef_list),
                 "name": np.asarray(geo_name_list)}
    if not os.path.exists(f"data/numpy_arrays/{anatomy}"):
        os.mkdir(f"data/numpy_arrays/{anatomy}")
    if not os.path.exists(f"data/numpy_arrays/{anatomy}/{set_type}"):
        os.mkdir(f"data/numpy_arrays/{anatomy}/{set_type}")
    save_dict(data_dict, f"data/numpy_arrays/{anatomy}/{set_type}/{anatomy}_{set_type}_data_dict")

    return

if __name__ == "__main__":
    anatomy = sys.argv[1]
    set_type = sys.argv[2]
    unsteady = bool(sys.argv[3])
    print("running function")
    get_numpy_arrays(anatomy, set_type, unsteady = False)