import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.simulation.get_avg_sol import *
from util.simulation.local.local_write_solver_files import *
from util.tools.junction_proc import *

def check_convergence(anatomy, geo_name, flow_index):
    flow_name = f"flow_{flow_index}"
    results_dir = f"/home/nrubio/Desktop/junction_pressure_differentials/data/synthetic_junctions_reduced_results/{anatomy}/{geo_name}/flow_{flow_index}_red_sol"
    centerline_dir = f"/home/nrubio/Desktop/junction_pressure_differentials/data/synthetic_junctions/{anatomy}/{geo_name}/centerlines/centerline.vtp"
    print("Averaging 3D results.")

    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = load_centerline_data(fpath_1d = centerline_dir)
    junction_dict, junc_pt_ids = identify_junctions_synthetic(junction_id, branch_id, pt_id)

    soln_dict, conv = get_avg_steady_results(ss_tol = 0.05,
                    fpath_1d = centerline_dir,
                    fpath_3d = f"data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}/solution_flow_{flow_index}.vtu",
                    fpath_out = results_dir,
                    pt_inds = junc_pt_ids, only_caps=False)


    if conv == True:
        print("Converged!")
        sys.exit(1)

    return

anatomy = sys.argv[1]; geo_name = sys.argv[2]; flow_index = sys.argv[3]
check_convergence(anatomy = anatomy, geo_name = geo_name, flow_index = flow_index)
