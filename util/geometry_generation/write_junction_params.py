import sys
sys.path.append("/Users/natalia/Desktop/junction_pressure_differentials")
from util.tools.basic import *

def generate_param_stat_dicts():
    # Distribution informed by Aorta and Pulmonary data
    anatomy = "AP"
    params_stat_dict = {
                "inlet_radius": [0.1, 0.85],
                "radius_ratio": [0.3, 1],
                "angle": [0, 0.7*180/np.pi]}
    if not os.path.exists("data"):
        os.mkdir("data/param_stat_dicts")
    if not os.path.exists("data/param_stat_dicts"):
        os.mkdir("data/param_stat_dicts")
    
    save_dict(params_stat_dict, f"data/param_stat_dicts/param_stat_dict_{anatomy}")
    return

def generate_mean_params(anatomy):
    params_stat_dict = load_dict(f"data/param_stat_dicts/param_stat_dict_{anatomy}")
    param_dict = {}

    # radii
    param_dict["inlet_radius"] = (params_stat_dict["inlet_radius"][0] + params_stat_dict["inlet_radius"][1])/2
    radius_ratio = (params_stat_dict["radius_ratio"][0] + params_stat_dict["radius_ratio"][1])/2
    sampled_outlet_radius = param_dict["inlet_radius"] * radius_ratio
    computed_outlet_radius = (param_dict["inlet_radius"]**3 - sampled_outlet_radius**3)**(1/3) # Murray's law
    param_dict["outlet1_radius"] = max(sampled_outlet_radius, computed_outlet_radius)
    param_dict["outlet2_radius"] = min(sampled_outlet_radius, computed_outlet_radius)

    # lengths
    inlet_length_factor = 10
    outlet_length_factor = 20
    param_dict["inlet_length"] = param_dict["inlet_radius"]*inlet_length_factor
    param_dict["outlet1_length"] = param_dict["outlet1_radius"]*outlet_length_factor
    param_dict["outlet2_length"] = param_dict["outlet2_radius"]*outlet_length_factor

    # angles
    param_dict["angle1"] = (params_stat_dict["angle"][0] + params_stat_dict["angle"][1])/2
    param_dict["angle2"] = (params_stat_dict["angle"][0] + params_stat_dict["angle"][1])/2

    param_dict["inlet_velocity"] = 180
    return param_dict

def generate_random_params(anatomy):
    params_stat_dict = load_dict(f"data/param_stat_dicts/param_stat_dict_{anatomy}")
    param_dict = {}

    # radii
    param_dict["inlet_radius"] = np.random.uniform(low = params_stat_dict["inlet_radius"][0], high = params_stat_dict["inlet_radius"][1])
    radius_ratio = np.random.uniform(low = params_stat_dict["radius_ratio"][0], high = params_stat_dict["radius_ratio"][1])
    sampled_outlet_radius = param_dict["inlet_radius"] * radius_ratio
    computed_outlet_radius = (param_dict["inlet_radius"]**3 - sampled_outlet_radius**3)**(1/3) # Murray's law
    param_dict["outlet1_radius"] = max(sampled_outlet_radius, computed_outlet_radius)
    param_dict["outlet2_radius"] = min(sampled_outlet_radius, computed_outlet_radius)

    # lengths
    inlet_length_factor = 10
    outlet_length_factor = 20
    param_dict["inlet_length"] = param_dict["inlet_radius"]*inlet_length_factor
    param_dict["outlet1_length"] = param_dict["outlet1_radius"]*outlet_length_factor
    param_dict["outlet2_length"] = param_dict["outlet2_radius"]*outlet_length_factor

    # angles
    param_dict["angle1"] = np.random.uniform(low = params_stat_dict["angle"][0], high = params_stat_dict["angle"][1])
    param_dict["angle2"] = np.random.uniform(low = params_stat_dict["angle"][0], high = params_stat_dict["angle"][1])

    param_dict["inlet_velocity"] = 180
    return param_dict

def write_anatomy_junctions(anatomy, set_type, num_junctions):

    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists(f"data/synthetic_junctions/{anatomy}"):
        os.mkdir(f"data/synthetic_junctions/{anatomy}")
    if not os.path.exists(f"data/synthetic_junctions/{anatomy}/{set_type}"):
        os.mkdir(f"data/synthetic_junctions/{anatomy}/{set_type}")

    for i in range(num_junctions):
        junction_name = f"{anatomy[0:2]}_{format(int(i), '03d')}"
        if os.path.exists(f"data/synthetic_junctions/{anatomy}/{set_type}/{junction_name}/junction_params_dict") == False:
                if not os.path.exists(f"data/synthetic_junctions/{anatomy}/{set_type}/{junction_name}"):
                    os.mkdir(f"data/synthetic_junctions/{anatomy}/{set_type}/{junction_name}")

                print(f"Generating {junction_name}")
                if set_type == "mesh_convergence":
                    params = generate_mean_params(anatomy)

                elif set_type == "random":
                     continue
                
                save_dict(params, f"data/synthetic_junctions/{anatomy}/{set_type}/{junction_name}/junction_params_dict")
    return
if __name__ == '__main__':
    generate_param_stat_dicts()
    write_anatomy_junctions("AP", "mesh_convergence", 10)
