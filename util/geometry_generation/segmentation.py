import os
import sys
import numpy as np
import sv
sys.path.append("/home/nrubio/Desktop/SV_scripts/geometry_generation") # need to append absolute path to directory where helper_functions.py is stored
from util.geometry_generation.helper_functions import *
sys.path.pop()


def get_inlet_segmentations(geo_params):
    """
    """
    num_pts = 9

    char_len = geo_params["inlet_radius"]
    char_len = 0.5310796510320017*15
    y_in = np.linspace(-char_len/2, 0, num_pts, endpoint = True)
    inlet_path_points_list = [[0.0, float(y), 0.0] for y in y_in]
    inlet_path = sv.pathplanning.Path()
    for point in inlet_path_points_list:
        inlet_path.add_control_point(point)
    inlet_path_points_list = inlet_path.get_control_points()
    inlet_path_curve_points = inlet_path.get_curve_points()


    segmentations = []

    for i in range(num_pts-1):
        contour = sv.segmentation.Circle(radius = geo_params["inlet_radius"],
                                    center = inlet_path_points_list[i],
                                    normal = inlet_path.get_curve_tangent(inlet_path_curve_points.index(inlet_path_points_list[i])))
        segmentations.append(contour)

    contour = sv.segmentation.Circle(radius = geo_params["inlet_radius"]*1,
                                center = inlet_path_points_list[-1],
                                normal = inlet_path.get_curve_tangent(inlet_path_curve_points.index(inlet_path_points_list[i])))

    segmentations.append(contour)


    return inlet_path, segmentations

def get_main_segmentations(geo_params):
    """
    """
    num_pts = 5

    #char_len = geo_params["inlet_radius"]*10
    char_len = 0.5310796510320017*10
    y_in = np.linspace(-char_len, 0, num_pts+1, endpoint = True)
    inlet_path_points_list = [[0.0, float(y), 0.0] for y in y_in]
    inlet_path = sv.pathplanning.Path()
    for point in inlet_path_points_list:
        inlet_path.add_control_point(point)
    inlet_path_points_list = inlet_path.get_control_points()
    inlet_path_curve_points = inlet_path.get_curve_points()


    r = np.linspace(0, char_len, num_pts+1, endpoint = True)[1:]
    theta = np.ones((num_pts,)) * geo_params["outlet1_angle"]
    theta[0] = geo_params["outlet1_angle"]/3
    theta[1] = geo_params["outlet1_angle"]*2/3
    theta = np.pi/2 + np.pi * theta / 180
    outlet1_x = r * np.cos(theta)
    outlet1_y = r * np.sin(theta)
    outlet1_y[2:] = outlet1_y[2:] +  geo_params["inlet_radius"]*1.3
    outlet1_path_points_list = [[float(outlet1_x[i]), float(outlet1_y[i]), 0.0] for i in range(num_pts)]
    outlet1_path = sv.pathplanning.Path()
    for point in outlet1_path_points_list:
        outlet1_path.add_control_point(point)
    outlet1_path_points_list = outlet1_path.get_control_points()
    outlet1_path_curve_points = outlet1_path.get_curve_points()

    main_path_points_list = inlet_path_points_list + outlet1_path_points_list
    main_path = sv.pathplanning.Path()
    for point in main_path_points_list:
        main_path.add_control_point(point)


    segmentations = []

    for i in range(num_pts):
        contour = sv.segmentation.Circle(radius = geo_params["inlet_radius"],
                                    center = inlet_path_points_list[i],
                                    normal = inlet_path.get_curve_tangent(inlet_path_curve_points.index(inlet_path_points_list[i])))
        segmentations.append(contour)

    contour = sv.segmentation.Circle(radius = geo_params["inlet_radius"],
                                center = inlet_path_points_list[-1],
                                normal = inlet_path.get_curve_tangent(inlet_path_curve_points.index(inlet_path_points_list[i])))
    segmentations.append(contour)

    for i in range(num_pts):
        contour = sv.segmentation.Circle(radius = geo_params["outlet1_radius"],
                                    center = outlet1_path_points_list[i],
                                    normal = outlet1_path.get_curve_tangent(outlet1_path_curve_points.index(outlet1_path_points_list[i])))
        segmentations.append(contour)

    return main_path, segmentations

def get_outlet_segmentations(geo_params):
    """
    """
    num_pts = 5
    char_len = geo_params["inlet_radius"]*10

    r = np.linspace(0, char_len, num_pts+1, endpoint = True)[1:]
    theta = np.ones((num_pts,)) * geo_params["outlet2_angle"]
    theta[0] = geo_params["outlet2_angle"]/3
    theta[1] = geo_params["outlet2_angle"]*2/3
    theta = np.pi/2 - np.pi * theta / 180
    outlet2_x = r * np.cos(theta)
    outlet2_y = r * np.sin(theta)
    outlet2_y[2:] = outlet2_y[2:] +  geo_params["inlet_radius"]*1.3
    outlet2_path_points_list = [[float(outlet2_x[i]), float(outlet2_y[i]), 0.0] for i in range(num_pts)]
    outlet2_path = sv.pathplanning.Path()
    for point in outlet2_path_points_list:
        outlet2_path.add_control_point(point)
    outlet2_path_points_list = outlet2_path.get_control_points()
    outlet2_path_curve_points = outlet2_path.get_curve_points()

    segmentations = []

    contour = sv.segmentation.Circle(radius = 0.9 * geo_params["inlet_radius"],
                                    center = [0.0, -1*geo_params["inlet_radius"], 0.0],
                                    normal = [0.0, 1.0, 0.0])
    segmentations.append(contour)

    contour = sv.segmentation.Circle(radius = 0.9 * geo_params["inlet_radius"],
                                    center = [0.0, 0.0, 0.0],
                                    normal = [0.0, 1.0, 0.0])
    segmentations.append(contour)

    for i in range(num_pts):
        contour = sv.segmentation.Circle(radius = geo_params["outlet2_radius"],
                                    center = outlet2_path_points_list[i],
                                    normal = outlet2_path.get_curve_tangent(outlet2_path_curve_points.index(outlet2_path_points_list[i])))
        segmentations.append(contour)

    return outlet2_path, segmentations

def get_u_segmentations(geo_params):
    """
    """
    num_pts = 10
    inset = 1
    #char_len = geo_params["inlet_radius"]*12
    char_len = 0.5310796510320017*20
    y_in = np.linspace(-char_len, 0, num_pts+1, endpoint = True)


    r = np.linspace(0, char_len, num_pts, endpoint = True)#[2:]
    theta = np.ones((num_pts,)) * geo_params["outlet1_angle"]
    # theta[0] = geo_params["outlet1_angle"]/3
    # theta[1] = geo_params["outlet1_angle"]*2/3
    theta = np.pi/2 + np.pi * theta / 180
    outlet1_x = r * np.cos(theta) #+  geo_params["inlet_radius"]/2
    outlet1_y = r * np.sin(theta)
    outlet1_y = outlet1_y #-  geo_params["inlet_radius"]*1.5
    outlet1_path_points_list = [[float(outlet1_x[i]), float(outlet1_y[i]), 0.0] for i in range(1, num_pts)]
    outlet1_path_points_list.reverse()


    r = np.linspace(0, char_len, num_pts, endpoint = True)#[2:]
    theta = np.ones((num_pts,)) * geo_params["outlet2_angle"]
    # theta[0] = geo_params["outlet2_angle"]/3
    # theta[1] = geo_params["outlet2_angle"]*2/3
    theta = np.pi/2 - np.pi * theta / 180
    outlet2_x = r * np.cos(theta) #-  geo_params["inlet_radius"]/2
    outlet2_y = r * np.sin(theta)
    outlet2_y = outlet2_y #- geo_params["inlet_radius"]*1.5
    outlet2_path_points_list = [[float(outlet2_x[i]), float(outlet2_y[i]), 0.0] for i in range(1, num_pts)]

    # outlet1_path_points_list.append([(float(outlet1_x[1]) + float(outlet2_x[1]))/2,
    #                              (float(outlet1_y[1]) + float(outlet2_y[1]))/2 - 0.5*geo_params["inlet_radius"], 0.0])
    outlet1_path_points_list.append([0.0, 0.0, 0.0])

    u_path = sv.pathplanning.Path()


    for i, point in enumerate(outlet1_path_points_list):
        if i == 0:
            u_path.add_control_point(point)
        else:
            u_path.add_control_point(point, 0)

    #u_path.add_control_point([0.0, 0.0, 0.0])
    # u_path.add_control_point([(float(outlet1_x[-1]) + float(outlet2_x[-1]))/2
    #                             (float(outlet1_y[-1] + float(outlet2_y[-1]))/2, 0.0])


    for point in outlet2_path_points_list:
        u_path.add_control_point(point, 0)

    u_path_points_list = u_path.get_control_points()
    u_path_curve_points = u_path.get_curve_points()


    segmentations = []

    for i in range(num_pts-1):
        contour = sv.segmentation.Circle(radius = geo_params["outlet2_radius"],
                                    center = u_path_points_list[i],
                                    normal = u_path.get_curve_tangent(u_path_curve_points.index(u_path_points_list[i])))
        segmentations.append(contour)

    r_top = geo_params["inlet_radius"]*1.4
    r_side = geo_params["inlet_radius"]
    r_bottom = geo_params["inlet_radius"]*1.8
    num_el_pts = 20
    contour_pts = []

    # for i in range(num_el_pts):
    #     p = r_top
    #     theta = (np.pi/2) * i /num_el_pts
    #     r = r_side + (r_top - r_side)*(1-np.cos((np.pi/2)* i/num_el_pts))
    #     contour_pts.append([0, r*np.sin(theta), r*np.cos(theta)])
    #
    # for i in range(num_el_pts):
    #     r = r_top - (r_top - r_side)*(1-np.cos((np.pi/2)* i/num_el_pts))
    #     theta = (np.pi/2) + (np.pi/2) * i /num_el_pts
    #     contour_pts.append([0, r*np.sin(theta), r*np.cos(theta)])
    #
    # for i in range(num_el_pts):
    #     r = r_side + (r_bottom - r_side)*(1-np.cos((np.pi/2)* i/num_el_pts))
    #     theta = (np.pi) + (np.pi/2) * i /num_el_pts
    #     contour_pts.append([0, r*np.sin(theta), r*np.cos(theta)])
    #
    # for i in range(num_el_pts):
    #     r = r_bottom - (r_bottom - r_side)*(1-np.cos((np.pi/2)* i/num_el_pts))
    #     theta = (3*np.pi/2) + (np.pi/2) * i /num_el_pts
    #     contour_pts.append([0, r*np.sin(theta), r*np.cos(theta)])

    for i in range(num_el_pts):
        z = r_side * (1-i /num_el_pts)
        y = r_top * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y, z])

    for i in range(num_el_pts):
        z = -r_side * (i /num_el_pts)
        y = r_top * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y, z])

    for i in range(num_el_pts):
        z = -r_side * (1- i /num_el_pts)
        y = -r_top * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y, z])

    for i in range(num_el_pts):
        z = r_side * (i /num_el_pts)
        y = -r_top * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y, z])

    contour = sv.segmentation.Contour(contour_pts)
    # contour = sv.segmentation.Circle(radius = geo_params["inlet_radius"]*1.2,
    #                             center = u_path_points_list[num_pts-1],
    #                             normal = u_path.get_curve_tangent(u_path_curve_points.index(u_path_points_list[num_pts-1])))
    segmentations.append(contour)

    for i in range(num_pts, 2*num_pts - 1):
        contour = sv.segmentation.Circle(radius = geo_params["outlet1_radius"],
                                    center = u_path_points_list[i],
                                    normal = u_path.get_curve_tangent(u_path_curve_points.index(u_path_points_list[i])))
        segmentations.append(contour)

    return u_path, segmentations



def get_junction_segmentation(geo_params):

    # main_vessel_path, main_vessel_segmentations = get_main_segmentations(geo_params)
    # main_vessel_polydata_objects = [contour.get_polydata() for contour in main_vessel_segmentations]
    #
    # outlet2_path, outlet2_segmentations = get_outlet_segmentations(geo_params)
    # outlet2_segmentations_polydata_objects = [contour.get_polydata() for contour in outlet2_segmentations]
    inlet_path, inlet_segmentations = get_inlet_segmentations(geo_params)
    inlet_segmentations_polydata_objects = [contour.get_polydata() for contour in inlet_segmentations]

    u_path, u_segmentations = get_u_segmentations(geo_params)
    u_segmentations_polydata_objects = [contour.get_polydata() for contour in u_segmentations]
    # sv.dmg.add_path(name = "main_vessel_path", path = main_vessel_path)
    # sv.dmg.add_path(name = "outlet2_path", path = outlet2_path)
    # sv.dmg.add_path(name = "inlet_path", path = inlet_path)
    # sv.dmg.add_path(name = "u_path", path = u_path)
    # # sv.dmg.add_segmentation(name = "main_vessel_segmentations", path = "main_vessel_path", segmentations = main_vessel_segmentations)
    # # sv.dmg.add_segmentation(name = "outlet2_segmentations", path = "outlet2_path", segmentations = outlet2_segmentations)
    # sv.dmg.add_segmentation(name = "inlet_segmentations", path = "inlet_path", segmentations = inlet_segmentations)
    # sv.dmg.add_segmentation(name = "u_segmentations", path = "u_path", segmentations = u_segmentations)

    return inlet_segmentations_polydata_objects, u_segmentations_polydata_objects


def get_pipe_segmentation(geo_params):

    inlet_path, inlet_segmentations = get_inlet_segmentations(geo_params)
    inlet_segmentations_polydata_objects = [contour.get_polydata() for contour in inlet_segmentations]

    return (inlet_segmentations_polydata_objects, )
