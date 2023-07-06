# source: email with Elena Martinez on 11/4/2020
import numpy as np
import sv

def create_path_from_points_list(path_points_list):
    """
    Purpose:
        Create an SV path object from the given list of path point coordinates.
    Inputs:
        list or np.array path_points_list
            = [list of coordinates of path points]
                where coordinates is a list of the form, [x, y, z]
    Returns:
        sv.pathplanning.Path() path
    """
    path = sv.pathplanning.Path()
    for point in path_points_list:
        path.add_control_point(point)
    return path

def create_circular_segmentations_from_path_and_radii_list(path, radii_list):
    """
    Purpose:
        Create a list of SV (circular) segmentation objects from the given SV path object and the list of radii prescribed at each path point.
    Inputs:
        sv.pathplanning.Path() path
        list or np.array path_points_list
            = [list of coordinates of path points]
                where coordinates is a list of the form, [x, y, z]
        list or np.array radii_list
            = [radius for each point in path_points_list]
    Returns:
        list segmentations
            = [list of sv.segmentation.Circle contour objects]
    """
    path_points_list = path.get_control_points()
    path_curve_points = path.get_curve_points() # path curve points are not the same as the path control points (used in function create_path_from_points_list). There are more path curve points than path control points. The path curve points are the points that comprise the path spline. However, the path control points are included in the path curve points.
    segmentations = []
    segmentations_polydata_objects = []
    for path_point_id in range(len(path_points_list)):
        contour = sv.segmentation.Circle(radius = radii_list[path_point_id], center = path_points_list[path_point_id], normal = path.get_curve_tangent(path_curve_points.index(path_points_list[path_point_id])))
        segmentations.append(contour)
        segmentations_polydata_objects.append(contour.get_polydata())
    return segmentations, segmentations_polydata_objects

def resample_and_align_segmentations(segmentations_polydata_objects):
    # Resample and align the contour polydata objects to ensure that all
    # contours contain the same quantity of points and are all rotated such that
    # the ids of each point in the contours are in the same position along the
    # contours for lofting.
    num_samples = 25    # Number of samples to take around circumference of contour.
    use_distance = True # Specify option for contour alignment.
    for index in range(0, len(segmentations_polydata_objects)):
        # Resample the current contour.
        segmentations_polydata_objects[index] = sv.geometry.interpolate_closed_curve(
                                            polydata = segmentations_polydata_objects[index],
                                            number_of_points = num_samples)

        # Align the current contour with the previous one, beginning with the
        # second contour.
        if index != 0:
            segmentations_polydata_objects[index] = sv.geometry.align_profile(
                                                segmentations_polydata_objects[index - 1],
                                                segmentations_polydata_objects[index],
                                                use_distance)
    return segmentations_polydata_objects

def get_center_segmentation_old(geo_params, smaller = False):

    rt1 = np.cos(np.pi * geo_params["outlet1_angle"]/180) * 2 * geo_params["outlet1_radius"]
    rt2 = np.cos(np.pi * geo_params["outlet2_angle"]/180) * 2 * geo_params["outlet2_radius"]


    rmax = max([geo_params["outlet1_radius"], geo_params["outlet2_radius"]])
    rmax = max([rmax, 0.5 * (rt1 + rt2)])
    theta = [0, np.pi/2, np.pi, 3*np.pi/2]
    r = [rt2, rmax, rt1, rmax]

    if smaller == True:
        r = [0.9*rval for rval in r]

    theta_list = list(theta)
    r_list = list(r)
    control_pts = []
    control_pts_lower = []
    for i in range(len(theta_list)):
        control_pts.append([r_list[i] * np.sin(theta_list[i]), r_list[i] * np.cos(theta_list[i]), 0])
        control_pts_lower.append([r_list[i] * np.sin(theta_list[i]), r_list[i] * np.cos(theta_list[i]), -0.1*rmax])

    contour = sv.segmentation.SplinePolygon(control_pts)
    contour_lower = sv.segmentation.SplinePolygon(control_pts_lower)

    return contour, contour_lower

def get_center_segmentation(geo_params, smaller = False):

    rt1 = np.cos(np.pi * geo_params["outlet1_angle"]/180) * 2 * geo_params["outlet1_radius"]
    rt2 = np.cos(np.pi * geo_params["outlet2_angle"]/180) * 2 * geo_params["outlet2_radius"]


    rmax = max([geo_params["outlet1_radius"], geo_params["outlet2_radius"]])
    rmax = max([rmax, 0.5 * (rt1 + rt2)])
    theta = [0, np.pi/2, np.pi, 3*np.pi/2]
    r = [rt2, rmax, rt1, rmax]

    if smaller == True:
        r = [0.9*rval for rval in r]

    theta_list = list(theta)
    r_list = list(r)
    control_pts = []
    control_pts_lower = []
    for i in range(len(theta_list)):
        control_pts.append([r_list[i] * np.sin(theta_list[i]), r_list[i] * np.cos(theta_list[i]), 0])
        control_pts_lower.append([r_list[i] * np.sin(theta_list[i]), r_list[i] * np.cos(theta_list[i]), -0.1*rmax])

    contour = sv.segmentation.SplinePolygon(control_pts)
    contour_lower = sv.segmentation.SplinePolygon(control_pts_lower)

    return contour, contour_lower
