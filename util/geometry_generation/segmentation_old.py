import os
import sys
import numpy as np
import sv
from util.geometry_generation.helper_functions import *



def get_main_vessel_segmentations(params):
    """
    Builds the main vessel (inlet and outlet1), which outlet 2 will branch off of
    """
    num_pts = 5
    main_path = sv.pathplanning.Path()

    # Add inlet points to path
    y_inlet = np.linspace(-1 * params["inlet_length"], 0, num_pts, endpoint = False)
    inlet_path_points_list = [[0.0, float(y), 0.0] for y in y_inlet]
    for point in inlet_path_points_list:
        main_path.add_control_point(point)


    # Add outlet 1 points to path
    t_outlet1 = np.linspace(0, params["outlet1_length"], num_pts + 1, endpoint = True)[1:]
    outlet1_path_points_list = [[-t*np.sin(np.pi * params["angle1"]/180), 
                                t*np.cos(np.pi * params["angle1"]/180),
                                0.0] for t in t_outlet1]
    for point in outlet1_path_points_list:
       main_path.add_control_point(point)

    main_path_points_list = main_path.get_control_points()
    main_path_curve_points = main_path.get_curve_points()

    # Get segmentations
    segmentations = []
    # Add inlet segmentations
    for i in range(num_pts):
        contour = sv.segmentation.Circle(radius = params["inlet_radius"],
                                    center = main_path_points_list[i],
                                    normal = main_path.get_curve_tangent(main_path_curve_points.index(main_path_points_list[i])))
        segmentations.append(contour)
    # Add outlet 1 segmentations
    for i in range(num_pts, 2*num_pts):
        contour = sv.segmentation.Circle(radius = params["outlet1_radius"],
                                    center = main_path_points_list[i],
                                    normal = main_path.get_curve_tangent(main_path_curve_points.index(main_path_points_list[i])))
        segmentations.append(contour)

    return main_path, segmentations

def get_branch_segmentations(params):
    """
    Builds the branch vessel
    """
    num_pts = 5
    branch_path = sv.pathplanning.Path()

    # Add outlet 2 points to path
    t_outlet2 = np.linspace(0, params["outlet2_length"]+2*params["outlet2_radius"]*1.5, num_pts + 1, endpoint = True)
    outlet2_path_points_list = [[t*np.sin(np.pi * params["angle2"]/180) - params["outlet1_radius"] * np.sin(np.pi * params["angle1"]/180), 
                                t*np.cos(np.pi * params["angle2"]/180),
                                0.0] for t in t_outlet2]
    for point in outlet2_path_points_list:
       branch_path.add_control_point(point)

    branch_path_points_list = branch_path.get_control_points()
    branch_path_curve_points = branch_path.get_curve_points()

    # Get segmentations
    segmentations = []
    # Add outlet 2 segmentations
    for i in range(num_pts):
        contour = sv.segmentation.Circle(radius = params["outlet2_radius"],
                                    center = branch_path_points_list[i],
                                    normal = branch_path.get_curve_tangent(branch_path_curve_points.index(branch_path_points_list[i])))
        segmentations.append(contour)

    return branch_path, segmentations


def get_junction_segmentations(params):
    main_path, main_segmentations = get_main_vessel_segmentations(params)
    main_segmentations_polydata_objects = [contour.get_polydata() for contour in main_segmentations]

    branch_path, branch_segmentations = get_branch_segmentations(params)
    branch_segmentations_polydata_objects = [contour.get_polydata() for contour in branch_segmentations]

    return main_segmentations_polydata_objects, branch_segmentations_polydata_objects


