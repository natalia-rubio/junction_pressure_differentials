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

def generate_junction_mesh(geo_name, geo_params, anatomy, mesh_divs):

    segmentations = get_junction_segmentation(geo_params)
    print("Segmentation Done!")
    model = construct_model(geo_name, segmentations, geo_params)
    print("Model Done!")
    mesh = get_mesh(geo_name, model, geo_params, anatomy, mesh_divs)
    print("Mesh Done!")
    return

def launch_anatomy_geo_sweep(anatomy, num_geos = 5):
    home_dir = os.path.expanduser("~")
    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    dir = "data/synthetic_junctions/"+anatomy
    geos = os.listdir(dir)
    for i in range(len(geos)):

        geo_name = geos[i]
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            #try:
            print("Generating Geometry %d"%i)
            print(dir+"/"+geo_name+"/junction_params_dict")
            geo_params = load_dict(dir+"/"+geo_name+"/junction_params_dict")

            print(geo_params)
            generate_junction_mesh(geo_name, geo_params, anatomy, 4)
            #
            # except:
            #     print("Failed mesh generation.")
            #     try:
            #         if os.path.exists(dir+"/"+geo_name+"/junction_params_dict"):
            #             os.remove(dir+"/"+geo_name+"/junction_params_dict")
            #         if os.path.exists(dir+"/"+geo_name):
            #             os.rmdir(dir+"/"+geo_name)
            #     except:
            #         continue

    return

def launch_mesh_sweep(anatomy, num_geos = 1):
    home_dir = os.path.expanduser("~")
    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    dir = "data/synthetic_junctions/"+anatomy
    geos = os.listdir(dir)
    mesh_divs_list = [1.5, 3, 6, 9, 12, 15, 18, 21]
    for i in range(len(geos)):

        geo_name = geos[i]
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            print("Generating Geometry %d"%i)
            print(dir+"/"+geo_name+"/junction_params_dict")
            geo_params = load_dict(dir+"/"+geo_name+"/junction_params_dict")

            print(geo_params)
            generate_mesh(geo_name, geo_params, anatomy, mesh_divs = mesh_divs_list[i])

    return

def write_pipe():
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
    launch_anatomy_geo_sweep(anatomy = "mynard_test", num_geos = 1)
    #write_pipe()
    #launch_mesh_sweep(anatomy = "mynard_vary_mesh_smooth", num_geos = 1)

# USE THIS COMMAND TO RUN WITH SIMVASCULAR:
# /usr/local/sv/simvascular/2023-02-02/simvascular --python -- util/geometry_generation/launch_anatomy_sweep.py
