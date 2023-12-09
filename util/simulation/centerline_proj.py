import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *

import vtk
import os
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm

from util.tools.get_bc_integrals import get_res_names
from util.tools.junction_proc import *
from util.tools.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import pickle
from sklearn.linear_model import LinearRegression

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def slice_vessel(inp_3d, origin, normal):
    """
    Slice 3d geometry at certain plane
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # cut 3d geometry
    cut_3d = cut_plane(inp_3d, origin, normal)
    #write_geo(f'slice_{origin[0]}.vtp', cut_3d.GetOutput())

    # extract region closest to centerline
    con = connectivity(cut_3d, origin)
    #write_geo(f'con_{origin[0]}.vtp', con.GetOutput())
    return con

def get_length(locs):
    length = 0
    for i in range(1, locs.shape[0]):
        length += np.linalg.norm(locs[i, :] - locs[i-1, :])
    return length

def get_integral(inp_3d, origin, normal):
    """
    Slice simulation at certain plane and integrate
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # slice vessel at given location
    inp = slice_vessel(inp_3d, origin, normal)

    # recursively add calculators for normal velocities
    for v in get_res_names(inp_3d, 'velocity'):
        fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        #fun = 'dot(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+',' + v + ")"
        inp = calculator(inp, fun, [v], 'normal_' + v)

    return Integration(inp)

def extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False):

    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()# get all result array names
    res_names = get_res_names(reader_3d, ['pressure', 'velocity'])# get point and normals from centerline
    points = v2n(reader_1d.GetPoints().GetData())
    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))# initialize output

    for name in res_names + ['area']:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
        array.Fill(0)
        reader_1d.GetPointData().AddArray(array) # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3 # integrate results on all points of intergration cells
    print(f"Extracting solution at {reader_1d.GetNumberOfPoints()} points.")
    for i in tqdm(range(reader_1d.GetNumberOfPoints())):
        # check if point is cap
        reader_1d.GetPointCells(i, ids)
        if ids.GetNumberOfIds() == 1:
            if gid[i] == 0:
                # inlet
                points[i] += eps_norm * normals[i]
            else:
                # outlets
                points[i] -= eps_norm * normals[i]
        else:
            if only_caps:
                continue # create integration object (slice geometry at point/normal)

        try:
            #import pdb; pdb.set_trace()
            integral = get_integral(reader_3d, points[i], normals[i])
        except Exception:
            continue # integrate all output arrays

        for name in res_names:
            reader_1d.GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
        reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())
    write_geo(fpath_out, reader_1d)
    #import pdb; pdb.set_trace()
    return


def plot_vars(anatomy, geometry, flow, plot_pressure = True):
    offset = 10
    fpath_1dsol = f"data/synthetic_junctions_reduced_results/{anatomy}/{geometry}/1dsol_flow_solution_{flow}.vtp"
    soln = read_geo(fpath_1dsol).GetOutput()  # get 3d flow data

    soln_array = get_all_arrays(soln)
    points = v2n(soln.GetPoints().GetData())
    #Extract Geometry ----------------------------------------------------
    pt_id = soln_array["GlobalNodeId"].astype(int)
    num_pts = np.size(pt_id)  # number of points in mesh
    branch_id = soln_array["BranchIdTmp"].astype(int)
    junction_id = soln_array["BifurcationId"].astype(int)
    inlet_pts = np.where(branch_id == 0)
    outlet1_pts = np.where(branch_id == 1)
    outlet2_pts = np.where(branch_id == 2)

    inlet_locs = (points[inlet_pts])[np.argsort(pt_id[inlet_pts])]
    inlet_length = get_length(inlet_locs[offset:])
    area_inlet = (soln_array["area"][inlet_pts])[np.argsort(pt_id[inlet_pts])][offset]
    q_inlet = (soln_array["velocity_01000"][inlet_pts])[np.argsort(pt_id[inlet_pts])][offset]
    p_inlet = (soln_array["pressure_01000"][inlet_pts])[np.argsort(pt_id[inlet_pts])] + 0.5*1.06*np.square(q_inlet/area_inlet)
    p_end_inlet = p_inlet[offset]

    inlet_inds = np.linspace(0, len(p_inlet), len(p_inlet))
    print(f"Inlet Pressure Difference: {np.max(p_inlet) - np.min(p_inlet)}")

    outlet1_locs = (points[outlet1_pts])[np.argsort(pt_id[outlet1_pts])]
    outlet1_length = get_length(outlet1_locs[:-offset])

    area_outlet1 = (soln_array["area"][outlet1_pts])[np.argsort(pt_id[outlet1_pts])][-offset]
    q_outlet1 = (soln_array["velocity_01000"][outlet1_pts])[np.argsort(pt_id[outlet1_pts])][-offset]
    p_outlet1 = (soln_array["pressure_01000"][outlet1_pts])[np.argsort(pt_id[outlet1_pts])] + 0.5*1.06*np.square(q_outlet1/area_outlet1)
    p_end_outlet1 = p_outlet1[-offset]
    print("Inlet flow")
    print(q_inlet)

    outlet1_inds = np.linspace(len(p_inlet), len(p_inlet)+len(p_outlet1), len(p_outlet1))

    outlet2_locs = (points[outlet2_pts])[np.argsort(pt_id[outlet2_pts])]
    outlet2_length = get_length(outlet2_locs[:-offset])


    area_outlet2 = (soln_array["area"][outlet2_pts])[np.argsort(pt_id[outlet2_pts])][-offset]
    q_outlet2 = (soln_array["velocity_01000"][outlet2_pts])[np.argsort(pt_id[outlet2_pts])][-offset]
    p_outlet2 = (soln_array["pressure_01000"][outlet2_pts])[np.argsort(pt_id[outlet2_pts])] + 0.5*1.06*np.square(q_outlet2/area_outlet2)
    p_end_outlet2 = p_outlet2[-offset]
    outlet2_inds = np.linspace(len(p_inlet), len(p_inlet)+len(p_outlet2), len(p_outlet2))

    reg_in = LinearRegression().fit(inlet_inds[2*offset: -offset].reshape(-1, 1), p_inlet[2*offset: -offset].reshape(-1, 1))
    p_inlet_pred  = reg_in.predict(inlet_inds.reshape(-1, 1))


    reg1 = LinearRegression().fit(outlet1_inds[-220: -offset].reshape(-1, 1), p_outlet1[-220: -offset].reshape(-1, 1))
    p_outlet1_pred  = reg1.predict(outlet1_inds.reshape(-1, 1))
    dp_junc1 = p_outlet1_pred[0] - p_inlet_pred[-1]

    reg2 = LinearRegression().fit(outlet2_inds[-220: -offset].reshape(-1, 1), p_outlet2[-220: -offset].reshape(-1, 1))
    p_outlet2_pred  = reg2.predict(outlet2_inds.reshape(-1, 1))
    dp_junc2 = p_outlet2_pred[0] - p_inlet_pred[-1]

    if plot_pressure:
        plt.clf()
        plt.plot(inlet_inds, p_inlet/1333, label = "Inlet", linewidth = 2, c = colors[0])
        plt.plot(inlet_inds, p_inlet_pred/1333, linewidth = 2, linestyle = "--", c = colors[0])
        plt.scatter(np.asarray([inlet_inds[offset],]), np.asarray([p_inlet[offset]/1333,]), marker = "o", c = colors[0])
        plt.plot(outlet1_inds, p_outlet1/1333, label = "Outlet 1", linewidth = 2, c = colors[1])
        plt.plot(outlet1_inds, p_outlet1_pred/1333, linewidth = 2, linestyle = "--", c = colors[1])
        plt.scatter(np.asarray(outlet1_inds[-offset]), np.asarray(p_outlet1[-offset]/1333), marker = "o", c = colors[1])
        plt.plot(outlet2_inds, p_outlet2/1333, label = "Outlet 2", linewidth = 2, c = colors[2])
        plt.plot(outlet2_inds, p_outlet2_pred/1333, linewidth = 2, linestyle = "--", c = colors[2])
        plt.scatter(np.asarray(outlet2_inds[-offset]), np.asarray(p_outlet2[-offset]/1333), marker = "o", c = colors[2])
        plt.xlabel("Centerline Distance")
        plt.ylabel("Average Pressure (mmHg)")
        plt.legend(fontsize="14")
        plt.savefig(f"{fpath_1dsol[0:-25]}/centerline_pressure_{fpath_1dsol[-5]}_extrap.pdf", bbox_inches='tight', format = "pdf")


    if plot_pressure:

        inlet_1d_incs = np.linalg.norm(inlet_locs[1:,:] - inlet_locs[0:-1, :], axis = 1)
        inlet_1d_locs = [0]
        for i in range(int(inlet_1d_incs.size)):
            inlet_1d_locs.append(inlet_1d_locs[i] + inlet_1d_incs[i])
        inlet_1d_locs = np.asarray(inlet_1d_locs)

        outlet1_1d_incs = np.linalg.norm(outlet1_locs[1:,:] - outlet1_locs[0:-1, :], axis = 1)
        outlet1_1d_locs = [inlet_1d_locs[-1]]
        for i in range(int(outlet1_1d_incs.size)):
            outlet1_1d_locs.append(outlet1_1d_locs[i] + outlet1_1d_incs[i])
        outlet1_1d_locs = np.asarray(outlet1_1d_locs)

        outlet2_1d_incs = np.linalg.norm(outlet2_locs[1:,:] - outlet2_locs[0:-1, :], axis = 1)
        outlet2_1d_locs = [inlet_1d_locs[-1]]
        for i in range(int(outlet2_1d_incs.size)):
            outlet2_1d_locs.append(outlet2_1d_locs[i] + outlet2_1d_incs[i])
        outlet2_1d_locs = np.asarray(outlet2_1d_locs)

        plt.clf()
        plt.plot(inlet_1d_locs, p_inlet/1333, label = "Inlet", linewidth = 2, c = colors[0])
        #plt.plot(inlet_inds, p_inlet_pred/1333, linewidth = 2, linestyle = "--", c = colors[0])
        plt.scatter(np.asarray([inlet_1d_locs[offset],]), np.asarray([p_inlet[offset]/1333,]), marker = "o", c = colors[0])
        plt.plot(outlet1_1d_locs, p_outlet1/1333, label = "Outlet 1", linewidth = 2, c = colors[1])
        #plt.plot(outlet1_inds, p_outlet1_pred/1333, linewidth = 2, linestyle = "--", c = colors[1])
        plt.scatter(np.asarray(outlet1_1d_locs[-offset]), np.asarray(p_outlet1[-offset]/1333), marker = "o", c = colors[1])
        plt.plot(outlet2_1d_locs, p_outlet2/1333, label = "Outlet 2", linewidth = 2, c = colors[2])
        #plt.plot(outlet2_inds, p_outlet2_pred/1333, linewidth = 2, linestyle = "--", c = colors[2])
        plt.scatter(np.asarray(outlet2_1d_locs[-offset]), np.asarray(p_outlet2[-offset]/1333), marker = "o", c = colors[2])
        plt.xlabel("Centerline Distance (cm)")
        plt.ylabel("Total Average Pressure (mmHg)")
        plt.legend(fontsize="14")
        plt.savefig(f"{fpath_1dsol[0:-25]}/centerline_pressure_{fpath_1dsol[-5]}_total.pdf", bbox_inches='tight', format = "pdf")

    print(f"AREA || inlet: {area_inlet}.  outlet_1: {area_outlet1}. outlet_2: {area_outlet2}")
    print(f"FLOW || inlet: {q_inlet}.  outlet_1: {q_outlet1}. outlet_2: {q_outlet2}")
    #print(f"PRESSURE || inlet: {p_inlet}.  outlet_1: {p_outlet1}. outlet_2: {p_outlet2}")
    #assert area_inlet > area_outlet1, "Outlet1 area larger than inlet area"
    #assert area_inlet > area_outlet2, "Outlet2 area larger than inlet area"
    assert (q_inlet - (q_outlet1 +  q_outlet2))/q_inlet < 0.02, "Flow not conserved"

    soln_dict = {"flow": np.asarray([q_outlet1, q_outlet2, q_inlet]),
                "dp_junc": np.asarray([dp_junc1[0], dp_junc2[0]]),
                "dp_end": np.asarray([p_end_outlet1 - p_end_inlet, p_end_outlet2 - p_end_inlet]),
                "area": np.asarray([area_outlet1, area_outlet2, area_inlet]),
                "length": np.asarray([outlet1_length, outlet2_length, inlet_length])}

    if soln_dict["area"][0] < soln_dict["area"][1]:
        print(f"Switching outlets on {geometry}.")
        for value in soln_dict.keys():

            tmp = copy.deepcopy(soln_dict[value][0])
            soln_dict[value][0] = soln_dict[value][1]
            soln_dict[value][1] = tmp


    save_dict(soln_dict, f"data/synthetic_junctions_reduced_results/{anatomy}/{geometry}/flow_{flow}_red_sol")
    return

anatomies = ["Aorta_vary_rout"]#["mynard_vary_mesh_red_ts", "Aorta_vary_rout"]

for anatomy in anatomies:
    geos = os.listdir(f"data/synthetic_junctions_reduced_results/{anatomy}")
    for geometry in geos:
        for flow in ["1",]:

            fpath_1d = f"data/synthetic_junctions/{anatomy}/{geometry}/centerlines/centerline.vtp"
            fpath_3d = f"data/synthetic_junctions/{anatomy}/{geometry}/flow_{flow}/solution_flow_{flow}.vtu"
            fpath_out = f"data/synthetic_junctions_reduced_results/{anatomy}/{geometry}/1dsol_flow_solution_{flow}.vtp"

            #extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False)
            #try:
            plot_vars(anatomy, geometry, flow)
            # except:
            #     continue
