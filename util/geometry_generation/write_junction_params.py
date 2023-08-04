import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *

def write_anatomy_junctions(anatomy, start, num_junctions):

    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    params_stat_dict = load_dict("data/param_stat_dict")["Aorta"]

    for i in range(start, num_junctions):
        junction_name = f"{anatomy[0:5]}_{i}"
        if os.path.exists(f"data/synthetic_junctions/{anatomy}/{junction_name}") == False:
                print(f"Generating {junction_name}")
                os.mkdir(f"data/synthetic_junctions/{anatomy}/{junction_name}")
                params_rand = write_junction_params_rand(params_stat_dict)
                save_dict(params_rand, f"data/synthetic_junctions/{anatomy}/{junction_name}/junction_params_dict")
    return

def write_rout_sweep_junctions(anatomy, start, num_junctions):

    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    params_stat_dict = load_dict("data/param_stat_dict")["Aorta"]

    for i in range(start, num_junctions):
        percentile = i / (num_junctions-1)
        print(percentile)
        junction_name = f"{anatomy[0:5]}_{i}"
        if os.path.exists(f"data/synthetic_junctions/{anatomy}/{junction_name}") == False:
                print(f"Generating {junction_name}")
                os.mkdir(f"data/synthetic_junctions/{anatomy}/{junction_name}")
                params_rand = write_junction_params_sweep_outlet_radius(params_stat_dict, percentile)
                save_dict(params_rand, f"data/synthetic_junctions/{anatomy}/{junction_name}/junction_params_dict")
    return

def write_angle_sweep_junctions(anatomy, start, num_junctions):

    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    params_stat_dict = load_dict("data/param_stat_dict")["Aorta"]

    for i in range(start, num_junctions):
        percentile = i / (num_junctions-1)
        print(percentile)
        junction_name = f"{anatomy[0:5]}_{i}"
        if os.path.exists(f"data/synthetic_junctions/{anatomy}/{junction_name}") == False:
                print(f"Generating {junction_name}")
                os.mkdir(f"data/synthetic_junctions/{anatomy}/{junction_name}")
                params_rand = write_junction_params_sweep_angle(params_stat_dict, percentile)
                save_dict(params_rand, f"data/synthetic_junctions/{anatomy}/{junction_name}/junction_params_dict")
    return

def write_mynard_junction():

    num_junctions = 1; anatomy = "mynard"
    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    params_stat_dict = load_dict("data/param_stat_dict")["Aorta"]

    for i in range(num_junctions):

        junction_name = f"mynard"
        if os.path.exists(f"data/synthetic_junctions/{anatomy}/{junction_name}") == False:
                print(f"Generating {junction_name}")
                os.mkdir(f"data/synthetic_junctions/{anatomy}/{junction_name}")
                params_rand = write_junction_params_mynard()
                save_dict(params_rand, f"data/synthetic_junctions/{anatomy}/{junction_name}/junction_params_dict")
    return

def get_random(stats_list):
    return stats_list[0] + (0.4 + 0.2 * np.random.default_rng(seed=None).random()) * (stats_list[1]-stats_list[0])

def get_mean(stats_list):
    return stats_list[0] + 0.5 * (stats_list[1]-stats_list[0])

def get_percentile(stats_list, percentile):

    return stats_list[0] + (stats_list[1]-stats_list[0])*0.4 + percentile * (stats_list[1]-stats_list[0])*0.2

def write_junction_params_rand(params_stat_dict):

    inlet_radius = get_random(params_stat_dict["inlet_radius"])
    outlet1_angle = get_random(params_stat_dict["angle"])*180/np.pi
    outlet2_angle = get_random(params_stat_dict["angle"])*180/np.pi

    outlet1_radius = inlet_radius * get_random(params_stat_dict["radius_ratio"])
    outlet2_radius = inlet_radius * get_random(params_stat_dict["radius_ratio"])

    inlet_flow = get_random(params_stat_dict["velocity"]) * np.pi * inlet_radius**2
    params_rand = {"inlet_radius" : inlet_radius,
                    "outlet1_radius": outlet1_radius,
                    "outlet2_radius": outlet2_radius,
                    "outlet1_angle": outlet1_angle,
                    "outlet2_angle": outlet2_angle,
                    "inlet_flow": inlet_flow}

    if params_rand["outlet1_radius"] < params_rand["outlet2_radius"]:
        tmp  = copy.deepcopy(params_rand["outlet1_radius"])
        params_rand["outlet1_radius"] = params_rand["outlet2_radius"]
        params_rand["outlet2_radius"] = tmp
        tmp  = copy.deepcopy(params_rand["outlet1_angle"])
        params_rand["outlet1_angle"] = params_rand["outlet2_angle"]
        params_rand["outlet2_angle"] = tmp


    return params_rand

def write_junction_params_sweep_outlet_radius(params_stat_dict, percentile):

    inlet_radius = get_mean(params_stat_dict["inlet_radius"])
    outlet1_angle = get_mean(params_stat_dict["angle"])*180/np.pi
    outlet2_angle = get_mean(params_stat_dict["angle"])*180/np.pi

    outlet1_radius = inlet_radius * get_mean(params_stat_dict["radius_ratio"])
    outlet2_radius = inlet_radius * get_percentile(params_stat_dict["radius_ratio"], percentile)

    inlet_flow = get_mean(params_stat_dict["velocity"]) * np.pi * inlet_radius**2

    params_rand = {"inlet_radius" : inlet_radius,
                    "outlet1_radius": outlet1_radius,
                    "outlet2_radius": outlet2_radius,
                    "outlet1_angle": outlet1_angle,
                    "outlet2_angle": outlet2_angle,
                    "inlet_flow": inlet_flow}
    if params_rand["outlet1_radius"] < params_rand["outlet2_radius"]:
        tmp  = copy.deepcopy(params_rand["outlet1_radius"])
        params_rand["outlet1_radius"] = params_rand["outlet2_radius"]
        params_rand["outlet2_radius"] = tmp
        tmp  = copy.deepcopy(params_rand["outlet1_angle"])
        params_rand["outlet1_angle"] = params_rand["outlet2_angle"]
        params_rand["outlet2_angle"] = tmp

    print(params_rand)
    return params_rand

def write_mynard_junction_params_sweep_outlet_radius():

    num_junctions = 1; anatomy = "mynard_test"

    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)

    for i in range(num_junctions):

        junction_name = f"mynard_{i}"
        if os.path.exists(f"data/synthetic_junctions/{anatomy}/{junction_name}") == False:
                print(f"Generating {junction_name}")
                os.mkdir(f"data/synthetic_junctions/{anatomy}/{junction_name}")
                params_rand = write_junction_params_mynard()
                params_rand["outlet2_radius"] = params_rand["outlet2_radius"] * (0.9 + i*0.05)


                if params_rand["outlet1_radius"] < params_rand["outlet2_radius"]:
                    tmp  = copy.deepcopy(params_rand["outlet1_radius"])
                    params_rand["outlet1_radius"] = params_rand["outlet2_radius"]
                    params_rand["outlet2_radius"] = tmp
                    tmp  = copy.deepcopy(params_rand["outlet1_angle"])
                    params_rand["outlet1_angle"] = params_rand["outlet2_angle"]
                    params_rand["outlet2_angle"] = tmp
                print(params_rand)
                save_dict(params_rand, f"data/synthetic_junctions/{anatomy}/{junction_name}/junction_params_dict")

    print(params_rand)
    return params_rand

def write_mynard_junction_params_sweep_mesh():

    num_junctions = 8; anatomy = "mynard_vary_mesh_smooth"

    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)

    for i in range(num_junctions):

        junction_name = f"mynard_{i}"
        if os.path.exists(f"data/synthetic_junctions/{anatomy}/{junction_name}") == False:
                print(f"Generating {junction_name}")
                os.mkdir(f"data/synthetic_junctions/{anatomy}/{junction_name}")
                params_rand = write_junction_params_mynard()

                if params_rand["outlet1_radius"] < params_rand["outlet2_radius"]:
                    tmp  = copy.deepcopy(params_rand["outlet1_radius"])
                    params_rand["outlet1_radius"] = params_rand["outlet2_radius"]
                    params_rand["outlet2_radius"] = tmp
                    tmp  = copy.deepcopy(params_rand["outlet1_angle"])
                    params_rand["outlet1_angle"] = params_rand["outlet2_angle"]
                    params_rand["outlet2_angle"] = tmp
                print(params_rand)
                save_dict(params_rand, f"data/synthetic_junctions/{anatomy}/{junction_name}/junction_params_dict")

    print(params_rand)
    return params_rand

def write_pipe_params_sweep_mesh():

    num_junctions = 1; anatomy = "pipe_bl_mid_short"

    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)

    for i in range(num_junctions):

        junction_name = f"pipe_{i}"
        if os.path.exists(f"data/synthetic_junctions/{anatomy}/{junction_name}") == False:
                print(f"Generating {junction_name}")
                os.mkdir(f"data/synthetic_junctions/{anatomy}/{junction_name}")
                params_rand = write_junction_params_mynard()

                if params_rand["outlet1_radius"] < params_rand["outlet2_radius"]:
                    tmp  = copy.deepcopy(params_rand["outlet1_radius"])
                    params_rand["outlet1_radius"] = params_rand["outlet2_radius"]
                    params_rand["outlet2_radius"] = tmp
                    tmp  = copy.deepcopy(params_rand["outlet1_angle"])
                    params_rand["outlet1_angle"] = params_rand["outlet2_angle"]
                    params_rand["outlet2_angle"] = tmp
                print(params_rand)
                save_dict(params_rand, f"data/synthetic_junctions/{anatomy}/{junction_name}/junction_params_dict")

    print(params_rand)
    return params_rand

def write_junction_params_sweep_angle(params_stat_dict, percentile):

    inlet_radius = get_mean(params_stat_dict["inlet_radius"])
    outlet1_angle = get_mean(params_stat_dict["angle"])*180/np.pi
    outlet2_angle = get_percentile(params_stat_dict["angle"], percentile)*180/np.pi

    outlet1_radius = inlet_radius * get_mean(params_stat_dict["radius_ratio"])
    outlet2_radius = inlet_radius * get_mean(params_stat_dict["radius_ratio"])

    inlet_flow = get_mean(params_stat_dict["velocity"]) * np.pi * inlet_radius**2

    params_rand = {"inlet_radius" : inlet_radius,
                    "outlet1_radius": outlet1_radius,
                    "outlet2_radius": outlet2_radius,
                    "outlet1_angle": outlet1_angle,
                    "outlet2_angle": outlet2_angle,
                    "inlet_flow": inlet_flow}
    print(params_rand)
    return params_rand

def write_junction_params_mynard():

    inlet_radius = 0.55
    outlet1_angle = 45
    outlet2_angle = 45

    outlet1_radius = 0.55
    outlet2_radius = 0.55

    re = 1800
    viscosity = 0.04
    density = 1.06
    U_in = re * viscosity / (density * 2 * inlet_radius)
    print(U_in)

    inlet_flow = (U_in/2) * (np.pi* inlet_radius**2)
    params_rand = {"inlet_radius" : inlet_radius,
                    "outlet1_radius": outlet1_radius,
                    "outlet2_radius": outlet2_radius,
                    "outlet1_angle": outlet1_angle,
                    "outlet2_angle": outlet2_angle,
                    "inlet_flow": inlet_flow}

    return params_rand

if __name__ == '__main__':
    #write_junction_params_mynard()
    #write_pipe_params_sweep_mesh()
    write_mynard_junction_params_sweep_outlet_radius()
    #write_rout_sweep_junctions(anatomy = "Aorta_vary_rout", start = 0, num_junctions = 5)
