import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *

def collect_synthetic_results_steady(anatomy, require4 = True):
    print(anatomy)

    char_val_dict = {"outlet_radius": [],
                    "inlet_outlet_area_rat": [],
                    "inlet_radius": [],
                    "outlet_area_rat": [],
                    "angle": [],
                    "flow_list": [],
                    "dP_list": [],
                    "name": [],
                    "outlet_area_total": []}

    home_dir = os.path.expanduser("~")
    geos = os.listdir(f"data/synthetic_junctions_reduced_results/{anatomy}"); geos.sort()

    for j, geo in enumerate(geos[0:]):

        try:
            junction_params = load_dict(f"data/synthetic_junctions/{anatomy}/{geo}/junction_params_dict")
            results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{geo}/"

            flow_lists = [[0],[0]]
            dP_lists = [[0],[0]]

            for i in range(4):
                try:
                    flow_result_dir = results_dir + f"flow_{i}_red_sol"

                    if not os.path.exists(results_dir):
                        if require4:
                            assert os.path.exists(results_dir), f"Flow {i} missing for geometry {geo}"
                        else:
                            continue
                    soln_dict = load_dict(flow_result_dir)
                    print(geo); print(soln_dict)

                    # Enforce that the first outlet always be the larger one!
                    if soln_dict["area"][0,1] < soln_dict["area"][0,2]:
                        print(f"Switching outlets on {geo}.")
                        for value in soln_dict.keys():
                            if value == "times":
                                continue
                            tmp = copy.deepcopy(soln_dict[value][:,1])
                            soln_dict[value][:,1] = soln_dict[value][:,2]
                            soln_dict[value][:,2] = tmp

                    if np.abs(np.sum(soln_dict["flow_in_time"][1, 1:]) - soln_dict["flow_in_time"][1,0])/soln_dict["flow_in_time"][1,0] > 0.01:
                        print(f"{geo} - mass not conserved.")
                        raise ValueError
                        continue


                    for outlet_ind in range(2):
                        flow_lists[outlet_ind] += [soln_dict["flow_in_time"][1, outlet_ind + 1]]
                        dP_lists[outlet_ind] += [soln_dict["pressure_in_time"][1, outlet_ind + 1] - soln_dict["pressure_in_time"][1, 0]]

                except:
                    print("Problem extracting simulation data.")
                    if require4:
                        raise(ValueError)
                    else:
                        continue
            if len(flow_lists[0]) == 1:
                continue
            char_val_dict["flow_list"] += flow_lists
            char_val_dict["dP_list"] += dP_lists
            char_val_dict["inlet_radius"] += [junction_params["inlet_radius"]**2, junction_params["inlet_radius"]**2]

            char_val_dict["outlet_radius"] += [junction_params["outlet1_radius"]**2, junction_params["outlet2_radius"]**2]
            char_val_dict["angle"] += [junction_params["outlet1_angle"], junction_params["outlet2_angle"]]
            char_val_dict["name"] += [geo+"_1", geo+"_2"]
        except:
            print(f"Problem extracting junction data.  Skipping {geo}.")
            continue


    save_dict(char_val_dict, f"data/characteristic_value_dictionaries/{anatomy}_synthetic_data_dict_steady")
    print(f"Extracted {len(char_val_dict['name'])} Outlets")
    return
