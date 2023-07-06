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

def generate_mesh(geo_name, geo_params, anatomy):

    segmentations = get_junction_segmentation(geo_params)
    print("Segmentation Done!")
    model = construct_model(geo_name, segmentations, geo_params)
    print("Model Done!")
    mesh = get_mesh(geo_name, model, geo_params, anatomy)
    print("Mesh Done!")
    return

def launch_anatomy_geo_sweep(anatomy, num_geos = 1):
    home_dir = os.path.expanduser("~")
    if not os.path.exists("data/synthetic_junctions"):
        os.mkdir("data/synthetic_junctions")
    if not os.path.exists("data/synthetic_junctions/"+anatomy):
        os.mkdir("data/synthetic_junctions/"+anatomy)
    dir = "data/synthetic_junctions/"+anatomy

    for i in range(num_geos):

        geo_name = "mynard"
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            #try:
            print("Generating Geometry %d"%i)
            print(dir+"/"+geo_name+"/junction_params_dict")
            geo_params = load_dict(dir+"/"+geo_name+"/junction_params_dict")

            print(geo_params)
            generate_mesh(geo_name, geo_params, anatomy)
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

if __name__ == "__main__":
    launch_anatomy_geo_sweep(anatomy = "mynard_over6_bl", num_geos = 1)

# USE THIS COMMAND TO RUN WITH SIMVASCULAR:
# /usr/local/sv/simvascular/2023-02-02/simvascular --python -- /home/nrubio/Desktop/SV_scripts/geometry_generation/launch_anatomy_sweep.py
