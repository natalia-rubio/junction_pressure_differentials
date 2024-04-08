import sys
sys.path.append("/Users/natalia/Desktop/junction_pressure_differentials")
from util.geometry_generation.segmentation import *
from util.geometry_generation.modeling_and_meshing import *
from util.geometry_generation.initialize_soln import *

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs, sphere_ref = 2):

    segmentations = get_junction_segmentations(geo_params)
    print("Segmentation Done!")
    model = construct_model(geo_name, segmentations, geo_params)
    print("Model Done!")
    print("Meshing divs:" , mesh_divs)
    mesh = get_mesh(geo_name, model, geo_params, anatomy, set_type, mesh_divs, sphere_ref)
    print("Mesh Done!")
    return

def launch_anatomy_geo_sweep(anatomy, set_type, num_geos = 5):

    dir = "data/synthetic_junctions/"+anatomy+"/"+set_type
    geos = os.listdir(dir); geos.sort()
    for i in range(200, num_geos):

        geo_name = geos[i]
        print(geo_name)
        if not geo_name[0].isalnum():
            continue
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete/mesh-surfaces"):
            if len(os.listdir(dir+"/"+geo_name+"/mesh-complete/mesh-surfaces")) != 3:
                os.system("rm -r " + dir+"/"+geo_name+"/mesh-complete")
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:
            print("Generating Geometry " + geo_name)
            #os.system("rm " + dir+"/"+geo_name+"/mesh-complete/model_tmp.vtp")
            print(dir+"/"+geo_name+"/junction_params_dict")
            geo_params = load_dict(dir+"/"+geo_name+"/junction_params_dict")

            print(geo_params)
            if anatomy == "AP":
                try:
                    generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs = 1.5)
                    generate_initial_sol(geo_name, anatomy, set_type, geo_params)
                except:
                    print("Problem construction geometry " + geo_name)
                    #pdb.set_trace()
                    continue
            else:
                print("Didn't recognize anatomy type.")
    return

def launch_mesh_sweep(anatomy, set_type, num_geos = 1):

    dir = "data/synthetic_junctions/" + anatomy + "/" + set_type
    geos = os.listdir(dir)
    geos.sort() 
    print(geos)
    for geo in geos:
        if not geo[0].isalnum():
            geos.remove(geo)
    mesh_divs_list_curved = [0.75, 1, 1.5, 2, 3, 4]
    sphere_ref_list = [1,2,3,4]
    for i in range(num_geos):#len(geos)):

        geo_name = geos[i]
        if not geo_name[0].isalnum():
            continue
        if os.path.exists(dir+"/"+geo_name+"/mesh-complete") == False:

            print("Generating Geometry " + geo_name)
            geo_params = load_dict(dir+"/"+geo_name+"/junction_params_dict")
            print(geo_params)
            if anatomy == "AP":
                print("i:", i)
                print("Mesh divs:", mesh_divs_list_curved[i])

                generate_vessel_mesh(geo_name, geo_params, anatomy, set_type, mesh_divs = mesh_divs_list_curved[i])
                generate_initial_sol(geo_name, anatomy, set_type, geo_params)
            else:
                print("Didn't recognize anatomy type.")

    return

def generate_geometries(anatomy, set_type, num_geos):
    if set_type == "mesh_convergence":
        launch_mesh_sweep(anatomy, set_type, num_geos)
    elif set_type == "random":
        launch_anatomy_geo_sweep(anatomy, set_type, num_geos)
    return

# if __name__ == "__main__":
#     generate_geometries(anatomy = "curved", set_type = "mesh_convergence", num_geos = 150)

if __name__ == "__main__":
    anatomy = sys.argv[1]
    set_type = sys.argv[2]
    generate_geometries(anatomy = anatomy, set_type = set_type, num_geos = 300)
    # geo_params = load_dict("/Users/natalia/Desktop/vessel_pressure_differentials/data/synthetic_junctions/test/test/vertical_working/vessel_params_dict")

    # generate_vessel_mesh("vertical_not_working", geo_params, "test", "test", mesh_divs = 2)

# USE THIS COMMAND TO RUN WITH SIMVASCULAR PYTHON:
# /usr/local/sv/simvascular/2023-02-02/simvascular --python -- util/geometry_generation/launch_anatomy_sweep.py
# mesh_convergence
#/Applications/SimVascular.app/Contents/Resources/simvascular --python -- util/geometry_generation/construct_geometries.py