import os
import sys
import numpy as np
import sv
sys.path.append("/home/nrubio/Desktop/SV_scripts/geometry_generation") # need to append absolute path to directory where helper_functions.py is stored
from util.geometry_generation.helper_functions import *
sys.path.pop()

def get_mynard_inlet_segmentations(geo_params):
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

    for i in range(num_pts-3):
        contour = sv.segmentation.Circle(radius = geo_params["inlet_radius"],
                                    center = inlet_path_points_list[i],
                                    normal = inlet_path.get_curve_tangent(inlet_path_curve_points.index(inlet_path_points_list[i])))
        segmentations.append(contour)

    r_top = geo_params["inlet_radius"]
    r_side = geo_params["inlet_radius"]
    r_bottom = geo_params["inlet_radius"]
    num_el_pts = 20
    contour_pts = []
    y = 0#inlet_path_points_list[-1][1]
    for i in range(num_el_pts):
        x = r_side * (1-i /num_el_pts)
        z = r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = -r_side * (i /num_el_pts)
        z = r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = -r_side * (1- i /num_el_pts)
        z = -r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = r_side * (i /num_el_pts)
        z = -r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    contour = sv.segmentation.Contour(contour_pts)
    segmentations.append(contour)
    return inlet_path, segmentations
def get_mynard_u_segmentations(geo_params):
    """
    """
    num_pts = 10
    inset = 1
    #char_len = geo_params["inlet_radius"]*12
    char_len = 0.5310796510320017*20
    y_in = np.linspace(-char_len, 0, num_pts+1, endpoint = True)


    r = np.linspace(0, char_len, num_pts, endpoint = True)#[2:]
    theta = np.ones((num_pts,)) * geo_params["outlet1_angle"]
    theta = np.pi/2 + np.pi * theta / 180
    outlet1_x = r * np.cos(theta)
    outlet1_y = r * np.sin(theta)
    outlet1_y = outlet1_y
    outlet1_path_points_list = [[float(outlet1_x[i]), float(outlet1_y[i]), 0.0] for i in range(1, num_pts)]
    outlet1_path_points_list.reverse()


    r = np.linspace(0, char_len, num_pts, endpoint = True)#[2:]
    theta = np.ones((num_pts,)) * geo_params["outlet2_angle"]
    theta = np.pi/2 - np.pi * theta / 180
    outlet2_x = r * np.cos(theta)
    outlet2_y = r * np.sin(theta)
    outlet2_y = outlet2_y
    outlet2_path_points_list = [[float(outlet2_x[i]), float(outlet2_y[i]), 0.0] for i in range(1, num_pts)]
    outlet1_path_points_list.append([0.0, 0.0, 0.0])

    u_path = sv.pathplanning.Path()


    for i, point in enumerate(outlet1_path_points_list):
        if i == 0:
            u_path.add_control_point(point)
        else:
            u_path.add_control_point(point, 0)


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

    r_base = max([geo_params["inlet_radius"], geo_params["outlet1_radius"], geo_params["outlet2_radius"]])
    print(r_base)
    r_top =  r_base*1.55
    r_side = r_base
    r_bottom = r_base*1.5
    num_el_pts = 20
    contour_pts = []

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
        y = -r_bottom * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y, z])

    for i in range(num_el_pts):
        z = r_side * (i /num_el_pts)
        y = -r_bottom * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y, z])

    contour = sv.segmentation.Contour(contour_pts)
    segmentations.append(contour)

    for i in range(num_pts, 2*num_pts - 1):
        contour = sv.segmentation.Circle(radius = geo_params["outlet1_radius"],
                                    center = u_path_points_list[i],
                                    normal = u_path.get_curve_tangent(u_path_curve_points.index(u_path_points_list[i])))
        segmentations.append(contour)

    return u_path, segmentations
def get_aorta_inlet_segmentations(geo_params):
    """
    """
    num_pts =10

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

    r_side = geo_params["inlet_radius"]
    r_top = (2*geo_params["inlet_radius"]+geo_params["outlet1_radius"])/3
    r_bottom = geo_params["inlet_radius"]
    num_el_pts = 20
    contour_pts = []
    y = 0.25#inlet_path_points_list[-1][1]
    for i in range(num_el_pts):
        x = r_side * (1-i /num_el_pts)
        z = r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = -r_side * (i /num_el_pts)
        z = r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = -r_side * (1- i /num_el_pts)
        z = -r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = r_side * (i /num_el_pts)
        z = -r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    contour = sv.segmentation.Contour(contour_pts)
    segmentations.append(contour)
    return inlet_path, segmentations
def get_pulmo_inlet_segmentations(geo_params):
    """
    """
    num_pts = 5

    char_len = geo_params["inlet_radius"]
    char_len = 0.23*15
    y_in = np.linspace(-char_len/2,0.0*geo_params["inlet_radius"], num_pts, endpoint = True)

    inlet_path_points_list = [[0.0, float(y), 0.0] for y in y_in]
    inlet_path = sv.pathplanning.Path()
    for point in inlet_path_points_list:
        inlet_path.add_control_point(point)
    inlet_path_points_list = inlet_path.get_control_points()
    inlet_path_curve_points = inlet_path.get_curve_points()


    segmentations = []

    for i in range(num_pts):
        contour = sv.segmentation.Circle(radius = geo_params["inlet_radius"],
                                    center = inlet_path_points_list[i],
                                    normal = inlet_path.get_curve_tangent(inlet_path_curve_points.index(inlet_path_points_list[i])))
        segmentations.append(contour)

    r_side = (3*geo_params["inlet_radius"]+0*geo_params["outlet1_radius"])/3
    r_top = (3*geo_params["inlet_radius"]+0*geo_params["outlet1_radius"])/3
    r_bottom = geo_params["inlet_radius"]
    num_el_pts = 20
    contour_pts = []
    y = 0 #inlet_path_points_list[-1][1] #geo_params["inlet_radius"]#
    for i in range(num_el_pts):
        x = r_side * (1-i /num_el_pts)
        z = r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = -r_side * (i /num_el_pts)
        z = r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = -r_side * (1- i /num_el_pts)
        z = -r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    for i in range(num_el_pts):
        x = r_side * (i /num_el_pts)
        z = -r_top * np.sqrt(1 - (x/r_side)**2)
        contour_pts.append([x, y, z])

    contour = sv.segmentation.Contour(contour_pts)
    #segmentations.append(contour)
    return inlet_path, segmentations

def get_pulmo_u_segmentations(geo_params):
    """
    """
    print(geo_params)
    num_pts = 4#10
    inset = 1
    #char_len = geo_params["inlet_radius"]*12
    char_len = 0.23*20
    y_in = np.linspace(-char_len, 0, num_pts+1, endpoint = True)


    r = np.linspace(0, char_len, num_pts, endpoint = True)#[2:]
    theta = np.ones((num_pts,)) * geo_params["outlet1_angle"]
    theta = np.pi/2 + np.pi * theta / 180
    outlet1_x = r * np.cos(theta) #+  geo_params["inlet_radius"]/2
    outlet1_y = r * np.sin(theta)
    outlet1_y = outlet1_y #-  geo_params["inlet_radius"]*1.5
    outlet1_path_points_list = [[float(outlet1_x[i]), float(outlet1_y[i]), 0.0] for i in range(1, num_pts)]
    outlet1_path_points_list.reverse()


    r = np.linspace(0, char_len, num_pts, endpoint = True)#[2:]
    theta = np.ones((num_pts,)) * geo_params["outlet2_angle"]
    theta = np.pi/2 - np.pi * theta / 180
    outlet2_x = r * np.cos(theta) #-  geo_params["inlet_radius"]/2
    outlet2_y = r * np.sin(theta)
    outlet2_y = outlet2_y #- geo_params["inlet_radius"]*1.5
    outlet2_path_points_list = [[float(outlet2_x[i]), float(outlet2_y[i]), 0.0] for i in range(1, num_pts)]

    outlet1_path_points_list.append([0.0, 0.0, 0.0])

    u_path = sv.pathplanning.Path()


    for i, point in enumerate(outlet1_path_points_list):
        if i == 0:
            u_path.add_control_point(point)
        else:
            u_path.add_control_point(point, 0)


    for point in outlet2_path_points_list:
        u_path.add_control_point(point, 0)

    u_path_points_list = u_path.get_control_points()
    u_path_curve_points = u_path.get_curve_points()


    segmentations = []

    for i in range(num_pts-1):
        contour = sv.segmentation.Circle(radius = geo_params["outlet2_radius"], #geo_params["outlet2_radius"],
                                    center = u_path_points_list[i],
                                    normal = u_path.get_curve_tangent(u_path_curve_points.index(u_path_points_list[i])))
        segmentations.append(contour)

    r_top = geo_params["inlet_radius"]*2.2+ geo_params["outlet1_radius"]
    r_side = geo_params["inlet_radius"]
    #r_side = geo_params["outlet2_radius"]*1.3
    r_bottom = geo_params["inlet_radius"]*2
    num_el_pts = 20
    contour_pts = []

    y_plus = 0# geo_params["inlet_radius"]/3
    for i in range(num_el_pts):
        z = r_side * (1-i /num_el_pts)
        y = r_top * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y + y_plus, z])

    for i in range(num_el_pts):
        z = -r_side * (i /num_el_pts)
        y = r_top * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y + y_plus, z])

    for i in range(num_el_pts):
        z = -r_side * (1- i /num_el_pts)
        y = -r_top * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y + y_plus, z])

    for i in range(num_el_pts):
        z = r_side * (i /num_el_pts)
        y = -r_top * np.sqrt(1 - (z/r_side)**2)
        contour_pts.append([0, y + y_plus, z])

    contour = sv.segmentation.Contour(contour_pts)
    segmentations.append(contour)

    for i in range(num_pts, 2*num_pts - 1):
        contour = sv.segmentation.Circle(radius = geo_params["outlet1_radius"], #geo_params["outlet1_radius"],
                                    center = u_path_points_list[i],
                                    normal = u_path.get_curve_tangent(u_path_curve_points.index(u_path_points_list[i])))
        segmentations.append(contour)

    return u_path, segmentations

def get_aorta_u_segmentations(geo_params):
    """
    """
    num_pts = 8#10
    inset = 1
    #char_len = geo_params["inlet_radius"]*12
    char_len = 0.5310796510320017*20
    y_in = np.linspace(-char_len, 0, num_pts+1, endpoint = True)


    r = np.linspace(0, char_len, num_pts, endpoint = True)#[2:]
    theta = np.ones((num_pts,)) * geo_params["outlet1_angle"]
    theta = np.pi/2 + np.pi * theta / 180
    outlet1_x = r * np.cos(theta) #+  geo_params["inlet_radius"]/2
    outlet1_y = r * np.sin(theta)
    outlet1_y = outlet1_y #-  geo_params["inlet_radius"]*1.5
    outlet1_path_points_list = [[float(outlet1_x[i]), float(outlet1_y[i]), 0.0] for i in range(1, num_pts)]
    outlet1_path_points_list.reverse()


    r = np.linspace(0, char_len, num_pts, endpoint = True)#[2:]
    theta = np.ones((num_pts,)) * geo_params["outlet2_angle"]
    theta = np.pi/2 - np.pi * theta / 180
    outlet2_x = r * np.cos(theta) #-  geo_params["inlet_radius"]/2
    outlet2_y = r * np.sin(theta)
    outlet2_y = outlet2_y #- geo_params["inlet_radius"]*1.5
    outlet2_path_points_list = [[float(outlet2_x[i]), float(outlet2_y[i]), 0.0] for i in range(1, num_pts)]

    outlet1_path_points_list.append([0.0, 0.0, 0.0])

    u_path = sv.pathplanning.Path()


    for i, point in enumerate(outlet1_path_points_list):
        if i == 0:
            u_path.add_control_point(point)
        else:
            u_path.add_control_point(point, 0)


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

    r_top = geo_params["inlet_radius"]*2.2
    r_side = geo_params["inlet_radius"]
    #r_side = geo_params["outlet2_radius"]*1.3
    r_bottom = geo_params["inlet_radius"]*2
    num_el_pts = 20
    contour_pts = []

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
    segmentations.append(contour)

    for i in range(num_pts, 2*num_pts - 1):
        contour = sv.segmentation.Circle(radius = geo_params["outlet1_radius"],
                                    center = u_path_points_list[i],
                                    normal = u_path.get_curve_tangent(u_path_curve_points.index(u_path_points_list[i])))
        segmentations.append(contour)

    return u_path, segmentations

def get_mynard_junction_segmentation(geo_params):
    inlet_path, inlet_segmentations = get_mynard_inlet_segmentations(geo_params)
    inlet_segmentations_polydata_objects = [contour.get_polydata() for contour in inlet_segmentations]

    u_path, u_segmentations = get_mynard_u_segmentations(geo_params)
    u_segmentations_polydata_objects = [contour.get_polydata() for contour in u_segmentations]

    return inlet_segmentations_polydata_objects, u_segmentations_polydata_objects

def get_aorta_junction_segmentation(geo_params):
    inlet_path, inlet_segmentations = get_aorta_inlet_segmentations(geo_params)
    inlet_segmentations_polydata_objects = [contour.get_polydata() for contour in inlet_segmentations]

    u_path, u_segmentations = get_aorta_u_segmentations(geo_params)
    u_segmentations_polydata_objects = [contour.get_polydata() for contour in u_segmentations]

    return inlet_segmentations_polydata_objects, u_segmentations_polydata_objects

def get_pulmo_junction_segmentation(geo_params):
    inlet_path, inlet_segmentations = get_pulmo_inlet_segmentations(geo_params)
    inlet_segmentations_polydata_objects = [contour.get_polydata() for contour in inlet_segmentations]

    u_path, u_segmentations = get_pulmo_u_segmentations(geo_params)
    u_segmentations_polydata_objects = [contour.get_polydata() for contour in u_segmentations]

    return [inlet_segmentations_polydata_objects, u_segmentations_polydata_objects]

def get_pipe_segmentation(geo_params):

    inlet_path, inlet_segmentations = get_inlet_segmentations(geo_params)
    inlet_segmentations_polydata_objects = [contour.get_polydata() for contour in inlet_segmentations]

    return (inlet_segmentations_polydata_objects, )
