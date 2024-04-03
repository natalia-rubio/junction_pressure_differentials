import os
import sys
import math
import numpy as np
import sv
import pickle
from sv import *
import vtk
import os
import platform
from util.geometry_generation.meshing_helpers import *
import copy
import pdb

def construct_model(model_name, segmentations, geo_params):
    center_shift = geo_params["outlet2_radius"] * np.cos(geo_params["angle2"] * np.pi/180)
    contour_list = segmentations
    capped_vessels = create_vessels(contour_list=contour_list)
    #unioned_model = capped_vessels[0]#
    unioned_model = union_all(capped_vessels)
    #unioned_model.write("junction_model_union", "vtp")
    model = clean(unioned_model)
    model.write("junction_model_cleaned", "vtp")
    model = norm(model)
    model.write("junction_model_normed", "vtp")
    #tmp = model.get_polydata()
    # smooth_model = model.get_polydata()
    # #smoothing_params = {'method':'constrained', 'num_iterations':100}

    print("Surface set.")
    face_angle = 10
    angle_inc = 10
    num_walls = 3
    num_caps = 6
    # while num_walls > 2: # or num_caps > 5:
        
    #     model.compute_boundary_faces(face_angle)
    #     caps = model.identify_caps()
    #     ids = model.get_face_ids()
    #     walls = [ids[i] for i,x in enumerate(caps) if not x]
    #     num_walls = len(walls)
    #     #num_caps
    #     #print(caps)
    #     print(num_walls)

    #     if num_walls < 2:
    #         face_angle -= angle_inc
    #         angle_inc *= 0.1
    #         num_walls = 3
            
    #     face_angle += angle_inc

    # assert num_walls == 2, "Model has more than 2 faces."
    #pdb.set_trace()

    #model = blend_walls(model, blend_radius = 2*geo_params["outlet2_radius"]) #=min([4*geo_params["outlet1_radius"],
                                                #8*geo_params["outlet2_radius"]]))
    smoothing_params = {'method':'constrained', 'num_iterations':90, 'constrain_factor':1, 'num_cg_solves':50}
    smooth_model = sv.geometry.local_sphere_smooth(surface = model.get_polydata(), 
                                                   radius = 1.5*geo_params["outlet1_radius"], 
                                                   center = [0, 0, 0], 
                                                   smoothing_parameters = smoothing_params)
    model.set_surface(smooth_model)

    smoothing_params = {'method':'constrained', 'num_iterations':10, 'constrain_factor':1, 'num_cg_solves':50}
    smooth_model = sv.geometry.local_sphere_smooth(surface = model.get_polydata(), 
                                                   radius = 3*geo_params["outlet1_radius"], 
                                                   center = [0, 0, 0], 
                                                   smoothing_parameters = smoothing_params)
    model.set_surface(smooth_model)

    model.write("blended_model", "vtp")
    model.compute_boundary_faces(85)
    model, walls, caps, ids = combine_walls(model)
    model = combine_caps(model, walls, ids, num_caps = 3)
    print("boundary faces computed")
    model.write("junction_model", "vtp")

    return model

def get_mesh(model_name, model, geo_params, anatomy, set_type, mesh_divs, sphere_ref):
    # center_shift = geo_params["outlet1_radius"] * np.cos(geo_params["angle1"] * np.pi/180)
    # #mesh_divs = 3
    sphere_ref = 0.2
    # sphere_offset = 0
    print("Mesh divs:", mesh_divs)

    edge_size = geo_params["outlet2_radius"]/mesh_divs #geo_params["outlet2_radius"]/500
    print("Edge size: ", edge_size)
    caps = model.identify_caps()
    ids = model.get_face_ids()
    walls = [ids[i] for i,x in enumerate(caps) if not x]
    faces = model.get_face_ids()
    cap_faces = [ids[i] for i,x in enumerate(caps) if x]
    print(cap_faces)
    # max_area_cap = get_inlet_cap(mesher, walls):
    #
    # out_caps = copy.copy(cap_faces)
    # out_caps.remove(max_area_cap)
    # centerlines = model.compute_centerlines(inlet_ids = [max_area_cap], outlet_ids = out_caps, use_face_ids = True)

    mesher = sv.meshing.create_mesher(sv.meshing.Kernel.TETGEN)
    mesher.set_model(model)
    mesher.set_boundary_layer_options(number_of_layers=3, edge_size_fraction=0.3, layer_decreasing_ratio=0.8, constant_thickness=False)
    #mesher.set_boundary_layer(number_of_layers = 4, edge_size_fraction = 0.5, layer_decreasing_ratio = 6,
    #   constant_thickness = False)
    options = sv.meshing.TetGenOptions(global_edge_size = edge_size, surface_mesh_flag=True, volume_mesh_flag=True)

    #
    options.sphere_refinement.append({'edge_size':edge_size*sphere_ref, 'radius':1.5*geo_params["outlet1_radius"], 
                                      'center':[0, 0, 0]})
    options.sphere_refinement_on = True

    #options.boundary_layer_inside = True

    options.optimization = 10
    options.quality_ratio = 1#1.4
    options.no_bisect = True
    options.minimum_dihedral_angle = 18.0

    print("Options values: ")
    [ print("  {0:s}:{1:s}".format(key,str(value))) for (key, value) in sorted(options.get_values().items()) ]

    mesher.set_walls(walls)
    mesher.generate_mesh(options)

    msh = mesher.get_mesh()
    generate_mesh_complete_folder(model, mesher, model_name, caps, ids, walls, faces, anatomy, set_type)
    return msh, model

def generate_mesh_complete_folder(model, mesher, model_name, caps, ids, walls, faces, anatomy, set_type):
    print("Generating mesh complete folder. \n")
    home_dir = os.path.expanduser("~")
    dir = "data/synthetic_junctions/" + anatomy + "/" + set_type + "/" + model_name

    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(dir + '/mesh-complete'):
        os.mkdir(dir +  '/mesh-complete')
    if not os.path.exists(dir + '/mesh-complete/mesh-surfaces'):
        os.mkdir(dir +  '/mesh-complete/mesh-surfaces')
    if not os.path.exists(dir + '/centerlines'):
        os.mkdir(dir +  '/centerlines')

    max_area_cap = get_inlet_cap(mesher, walls)
    print("Max Area Cap: ")
    print(max_area_cap)
    out_caps = faces[1:]
    out_caps.remove(max_area_cap)
    assert len(out_caps) == 3, "Wrong number of caps."
    np.save(dir+"/max_area_cap", np.asarray([max_area_cap]), allow_pickle = True)

    model.write(dir +  '/mesh-complete'+os.sep+'model_tmp','vtp')
    mesher.write_mesh(dir +  '/mesh-complete'+os.sep+'mesh-complete.mesh.vtu')
    mesh_out = modeling.PolyData()
    mesh_out.set_surface(mesher.get_surface())
    mesh_out.write(dir +  '/mesh-complete'+os.sep+'mesh-complete','exterior.vtp')
    mesh_out.set_surface(mesher.get_face_polydata(walls[0]))
    mesh_out.write(dir +  '/mesh-complete'+os.sep+'walls_combined','vtp')
    for face in mesher.get_model_face_ids():
        if face == walls[0]:
            continue
        mesh_out.set_surface(mesher.get_face_polydata(face))
        mesh_out.write(dir +  '/mesh-complete/mesh-surfaces'+os.sep+'cap_{}'.format(face),'vtp')

    cent_solid = modeling.PolyData()
    cent = vmtk.centerlines(model.get_polydata(), inlet_ids = [max_area_cap], outlet_ids = out_caps, use_face_ids = True)
    print("Centerlines generated.")
    cent_solid.set_surface(cent)
    print("Centerline surface set.")
    cent_solid.write(dir +  '/centerlines/centerline', "vtp")
    return
