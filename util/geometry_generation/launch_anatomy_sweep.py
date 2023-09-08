import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
#from util.tools.basic import *
from util.geometry_generation.helper_functions import *
from util.geometry_generation.segmentation import *
from util.geometry_generation.modeling_and_meshing import *
import pickle
def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def generate_pipe_mesh(geo_name, geo_params, anatomy, mesh_divs):

    segmentations = get_pipe_segmentation(geo_params)
    print("Segmentation Done!")
    model = construct_model(geo_name, segmentations, geo_params)
    print("Model Done!")
    mesh = get_mesh(geo_name, model, geo_params, anatomy, mesh_divs)
    print("Mesh Done!")
    return

def generate_aorta_junction_mesh(geo_name, geo_params, anatomy, mesh_divs, sphere_ref, sphere_offset = 0):

    segmentations = get_aorta_junction_segmentation(geo_params)
    print("Segmentation Done!")
    model = construct_model(geo_name, segmentations, geo_params)
    print("Model Done!")
    mesh = get_mesh(geo_name, model, geo_params, anatomy, mesh_divs, sphere_ref, sphere_offset)
    print("Mesh Done!")
    return

def generate_mynard_junction_mesh(geo_name, geo_params, anatomy, mesh_divs, sphere_ref, sphere_offset = 0):

    segmentations = get_mynard_junction_segmentation(geo_params)
    print("Segmentation Done!")
    model = construct_model(geo_name, segmentations, geo_params)
    print("Model Done!")
    mesh = get_mesh(geo_name, model, geo_params, anatomy, mesh_divs, sphere_ref, sphere_offset)
    print("Mesh Done!")
    return

def launch_anatomy_geo_sweep(anatomy, num_geos = 5, anatomy_type = "mynard"):
    home_dir = os.path.expanduser("~")
    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    dir = "data/synthetic_junctions/"+anatomy
    geos = os.listdir(dir)
    for i in range(num_geos):

        geo_name = geos[i]
        print(geo_name)
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            try:
                print("Generating Geometry " + geo_name)
                print(dir+"/"+geo_name+"/junction_params_dict")
                geo_params = load_dict(dir+"/"+geo_name+"/junction_params_dict")

                print(geo_params)
                if anatomy_type == "mynard":
                    generate_mynard_junction_mesh(geo_name, geo_params, anatomy, mesh_divs = 3, sphere_ref =0.5)
                elif anatomy_type == "Aorta":
                    generate_aorta_junction_mesh(geo_name, geo_params, anatomy, mesh_divs = 2.5, sphere_ref = 0.5, sphere_offset = 0.7)
                else:
                    print("Didn't recognize anatomy type.")

            except:
                print("Failed mesh generation.")
                try:
                    if os.path.exists(dir+"/"+geo_name+"/junction_params_dict"):
                        os.remove(dir+"/"+geo_name+"/junction_params_dict")
                    if os.path.exists(dir+"/"+geo_name):
                        os.rmdir(dir+"/"+geo_name)
                except:
                    continue

    return

def launch_mesh_sweep(anatomy, num_geos = 1, anatomy_type = "Aorta"):
    home_dir = os.path.expanduser("~")
    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    dir = "data/synthetic_junctions/"+anatomy
    geos = os.listdir(dir)
    geos.sort()
    print(geos)
    #mesh_divs_list = [0.5, 0.35, 0.2, 0.1, 0.09, 0.15]
    #mesh_divs_list = [1,  0.45, 0.2, 0.15]
    #mesh_divs_list = [0.8, 0.6, 0.5, 0.4]
    mesh_divs_list = [1.5, 2, 2.5, 3, 1.2, 1]
    for i in range(len(geos)):

        geo_name = geos[i]
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            print("Generating Geometry %d"%i)
            print(dir+"/"+geo_name+"/junction_params_dict")
            geo_params = load_dict(dir+"/"+geo_name+"/junction_params_dict")

            print(geo_params)
            if anatomy_type == "mynard":
                generate_mynard_junction_mesh(geo_name, geo_params, anatomy, mesh_divs = mesh_divs_list[i], sphere_ref = 0.5)
            elif anatomy_type == "Aorta":
                generate_aorta_junction_mesh(geo_name, geo_params, anatomy, mesh_divs = mesh_divs_list[i],  sphere_ref = 0.5, sphere_offset = 0.7)
            else:
                print("Didn't recognize anatomy type.")

    return

def generate_pipe():
    anatomy = "pipe_bl_mid_short"
    home_dir = os.path.expanduser("~")
    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    dir = "data/synthetic_junctions/"+anatomy
    geos = os.listdir(dir)
    print(geos)
    mesh_divs_list = [4, ]
    for i in range(len(geos)):

        geo_name = geos[i]
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            print("Generating Geometry %d"%i)
            print(dir+"/"+geo_name+"/junction_params_dict")
            geo_params = load_dict(dir+"/"+geo_name+"/junction_params_dict")

            print(geo_params)
            generate_pipe_mesh(geo_name, geo_params, anatomy, mesh_divs = mesh_divs_list[i])

    return

if __name__ == "__main__":
    launch_anatomy_geo_sweep(anatomy = "Aorta_rand", num_geos = 200, anatomy_type = "Aorta")
    #write_pipe()
    #launch_mesh_sweep(anatomy = "mynard_vary_mesh", num_geos = 4, anatomy_type = "mynard")
    #launch_mesh_sweep(anatomy = "Aorta_vary_mesh", num_geos = 6, anatomy_type = "Aorta")

# USE THIS COMMAND TO RUN WITH SIMVASCULAR:
# /usr/local/sv/simvascular/2023-02-02/simvascular --python -- util/geometry_generation/launch_anatomy_sweep.py
