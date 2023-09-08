import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.junction_proc import *
from util.tools.vtk_functions import *
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from util.junction_extraction_util.graph_handling import *
def print_stats(char_val_dict, anatomy, value):
    dat = np.asarray(char_val_dict[anatomy][value])
    print(f"{anatomy} {value} statistics:")

    print(f"Min: {np.min(dat)}.  Max {np.max(dat)}.")
    print(f"Mean: {np.mean(dat)}.  Standard Deviation {np.std(dat)}.")
    print("---------------------------------------------------------")
    return

def extract_characteristic_values():
    """
    Compile list of junction graphs (from synthetic data)
    """

    # relevant file paths:
    results_dir = "data/characteristic_value_dictionaries/vmr_char_val_dict"
    soln_dir = "/home/nrubio/Desktop/junction_ml/data/3d_flow_repository"
    centerline_dir = "/home/nrubio/Desktop/junction_ml/data/centerlines"
    anatomy_map_loc = "/home/nrubio/Desktop/junction_ml/data/model_labels_osmsc.csv"

    anatomy_map = np.genfromtxt(anatomy_map_loc, delimiter = "," , skip_header = 1, usecols = (0, 3), dtype = "str")
    char_val_dict = {}

    with open(f"{soln_dir}/filelist.txt") as f:
        content = f.readlines() # load in models from repository
    models = [x.strip() for x in content].copy()

    for model in models:

        anatomy = anatomy_map[anatomy_map[:,0] == model[:9],1][0]
        print(f"Model {model} ({anatomy})")

        if anatomy not in char_val_dict.keys():
            char_val_dict.update({anatomy: {"radius": [],
                                            "inlet_radius": [],
                                            "angle": [],
                                            "flow": [],
                                            "inlet_flow": [],
                                            "dP": [],
                                            "P": [],
                                            "name": []}})

        try:
            pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, pressure_in_time, flow_in_time, times, time_interval = \
            load_vmr_model_data(model, soln_dir)
            junction_dict = identify_junctions(junction_id, branch_id, pt_id)
            #
        except: print("Geometry Error."); continue

        try:
            pressure_in_time_aug, pressure_in_time_aug_der, pressure_in_time_aug_der2,\
            flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2, num_time_steps_model = process_soln(flow_in_time, pressure_in_time, times)

            for junction_id in junction_dict.keys():
                #import pdb; pdb.set_trace()
                max_flow_ind = np.argmax(flow_in_time_aug[:,np.argmax(np.abs(flow_in_time_aug[:, get_inds(arr = pt_id, vals = junction_dict[junction_id])]))])


                flow, flow_hist1, flow_hist2, flow_hist3, flow_der, flow_der2, pressure, pressure_der = get_soln_at_time(\
                flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2, pressure_in_time_aug, pressure_in_time_aug_der, max_flow_ind)
                if verify_bifurcation(flow, area, pressure = pressure, junc_pts = junction_dict[junction_id], pt_arr = pt_id) == False: continue
                inlets, outlets = classify_branches(flow, junc_pts = junction_dict[junction_id], pt_arr = pt_id)
                inlet_pts = get_inds(arr = pt_id, vals = inlets); outlet_pts = get_inds(arr = pt_id, vals = outlets)

                inlet_angles = np.asarray([angle1[inlet_pts],
                                        angle2[inlet_pts],
                                        angle3[inlet_pts]])

                outlet_angles = np.asarray([angle1[outlet_pts],
                                        angle2[outlet_pts],
                                        angle3[outlet_pts]])

                angle_diffs = get_angle_diff(inlet_angles, outlet_angles)/np.pi
                angle_diffs = angle_diffs.reshape((len(outlet_pts),))


                flow = list(flow_in_time_aug[max_flow_ind, outlet_pts])
                inlet_flow = list(flow_in_time_aug[max_flow_ind, inlet_pts])+list(flow_in_time_aug[max_flow_ind, inlet_pts])
                p = list(pressure_in_time_aug[max_flow_ind, outlet_pts]*0 + pressure_in_time_aug[max_flow_ind, outlet_pts])
                dp = list(pressure_in_time_aug[max_flow_ind, outlet_pts] - pressure_in_time_aug[max_flow_ind, inlet_pts])
                radius = list(np.sqrt(area[outlet_pts]/np.pi))
                inlet_radius = list(np.sqrt(area[inlet_pts]/np.pi))+list(np.sqrt(area[inlet_pts]/np.pi))
                #import pdb; pdb.set_trace()
                if anatomy == "Coronary" and (radius[0] > 1 or radius[1] > 1):
                    print("Skipping!")
                    continue
                char_val_dict[anatomy]["flow"] += flow
                char_val_dict[anatomy]["inlet_flow"] += inlet_flow
                char_val_dict[anatomy]["P"] += p
                char_val_dict[anatomy]["dP"] += dp
                char_val_dict[anatomy]["radius"] += radius
                char_val_dict[anatomy]["inlet_radius"] += inlet_radius
                char_val_dict[anatomy]["angle"] += list(angle_diffs)
                char_val_dict[anatomy]["name"] += 2 * [f"model_{model}_junction_{junction_id}"]
        except:
            continue
    save_dict(char_val_dict, results_dir)
    return

def print_aorta_study():

    anatomy = "Aorta"
    char_val_dict = load_dict("data/characteristic_value_dictionaries/vmr_char_val_dict")
    param_stat_dict = {}

    for anatomy in char_val_dict.keys():
        char_val_dict[anatomy].update({"velocity": np.asarray(char_val_dict[anatomy]["flow"])/ (np.pi*np.square(char_val_dict[anatomy]["radius"]))})
        char_val_dict[anatomy].update({"inlet_velocity": np.asarray(char_val_dict[anatomy]["inlet_flow"])/ (np.pi*np.square(char_val_dict[anatomy]["inlet_radius"]))})
        char_val_dict[anatomy].update({"radius_ratio": np.asarray(char_val_dict[anatomy]["radius"])/ np.asarray(char_val_dict[anatomy]["inlet_radius"])})
        value_list = ["flow", "angle", "inlet_radius", "velocity", "radius_ratio", "inlet_velocity"]

        param_stat_dict.update({anatomy:{}})

        for value in value_list:
            dat = np.asarray(char_val_dict[anatomy][value])
            print_stats(char_val_dict, anatomy, value)
            param_stat_dict[anatomy].update({value: [np.min(dat),
                                            np.max(dat),
                                            np.mean(dat),
                                            np.std(dat)]})

    print(param_stat_dict)
    params_stat_dir = "data/param_stat_dict"
    save_dict(param_stat_dict, params_stat_dir)

    return
if __name__ == '__main__':
    extract_characteristic_values()
    print_aorta_study()
