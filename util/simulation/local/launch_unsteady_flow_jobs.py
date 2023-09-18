import os
import sys
import numpy as np
import time
import copy
import pickle
sys.path.append("/home/users/nrubio/SV_scripts") # need tofrom write_solver_files import *
from write_solver_files import *
import subprocess
import time
import copy
from get_scalers_synthetic import *

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_cap_numbers(cap_dir):
    file_names = os.listdir(cap_dir)
    cap_numbers = []
    print(file_names)
    print(cap_dir)
    for cap_file in file_names:
        if cap_file[0:3] == "cap":
            cap_numbers.append(int(cap_file[4:-4]))
    return cap_numbers

def write_geo_job_unsteady(anatomy, geo, flow_name, flow_index, num_cores, num_time_steps):
    flow_name = "unsteady"
    geo = geo_name
    geo_job_script = f"#!/bin/bash\n\
# Name of your job\n\
#SBATCH --job-name={geo_name}_flow_sweep\n\
# Name of partition\n\
#SBATCH --partition=amarsden\n\
#SBATCH --output=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.o%j\n\
#SBATCH --error=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.e%j\n\
# The walltime you require for your simulation\n\
#SBATCH --time=01:00:00\n\
# Amount of memory you require per node. The default is 4000 MB (or 4 GB) per node\n\
#SBATCH --mem=10000\n\
#SBATCH --nodes=1\n\
#SBATCH --tasks-per-node={num_cores}\n\
# Load Modules\n\
module purge\n\
module load openmpi\n\
module load openblas\n\
module load system\n\
module load x11\n\
module load mesa\n\
module load gcc\n\
module load valgrind\n\
module load python/3.9.0\n\
module load py-numpy/1.20.3_py39\n\
module load py-scipy/1.6.3_py39\n\
module load gcc/10.1.0\n\
# Name of the executable you want to run\n\
source /home/users/nrubio/junctionenv/bin/activate\n\
/home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpre.exe /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_job.svpre\n\
echo 'Done with svPre.'\n\
conv=false\n\
conv_attempts=1\n\
indir='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}/24-procs_case'\n\
outdir='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}'\n\
echo 'Launching Simulation'\n\
cd /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo}/{flow_name}\n\
mpirun -n {num_cores} /home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svsolver-openmpi.exe {flow_name}_solver.inp\n\
echo 'Simulation completed'\n\
/home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpost.exe -vtkcombo -start 0 -stop {num_time_steps} -incr 1 -indir $indir -outdir $outdir -vtu solution_{flow_name}.vtu > /dev/null\n\
python3 /home/users/nrubio/SV_scripts/for_sherlock/extract_unsteady_results.py {geo_name} {anatomy}\n\
rmkk -r $outdir"
    f = open(f"/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.sh", "w")
    f.write(geo_job_script)
    f.close()
    return


anatomy = sys.argv[1]; num_time_steps = int(sys.argv[2]); num_cores = int(sys.argv[3]); num_geos = int(sys.argv[4]);

num_launched = 0
print(f"Launching {num_geos} unsteady flow sweeps.")
dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}"
geos = os.listdir(dir); geos.sort(); geo_ind = 0; #geos = ["Aorta_1_coarse"]

while num_launched < num_geos:

    geo = geos[geo_ind]; geo_name = geo; print(geo_name)
    geo_ind += 1
    if geo[0] != "A":
        print("Not an aorta.");# continue

    already_done = True
    if not os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{geo}/unsteady_red_sol"):
            already_done = False
    if already_done: print("Unstady simulation complete"); continue

    try:
        centerline_dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/centerlines/centerline.vtp"
        pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = load_centerline_data(fpath_1d = centerline_dir)
        junction_dict, junc_pt_ids = identify_junctions(junction_id, branch_id, pt_id)
    except:
        print(f"Couldn't process centerline.  Removing {geo_name}")
        os.system(f"rm -r /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}")
        continue

    try:
        params_dict = load_dict(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/junction_params_dict")
        dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}"
        inlet_flow = params_dict["inlet_flow"]
        inlet_area = np.pi * params_dict["inlet_radius"]**2

        inlet_cap_number = int(np.load(f"{dir}/{geo_name}/max_area_cap.npy")[0])
        cap_numbers = get_cap_numbers(f"{dir}/{geo_name}/mesh-complete/mesh-surfaces/"); print(cap_numbers)

        if len(cap_numbers) != 3:
            print("Wrong number of caps.  Deleting.")
            os.system(f"rm -r /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}")
    except:
        print(f"Couldn't find parameter dictionary.")
        continue

    try:
        flow_name = f"unsteady"
        if os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{geo_name}/{flow_name}_red_sol"):
            print(f"Simulation already complete for flow {flow_index}")
            continue

        flow_params = {"flow_amp": inlet_flow*1.2,
                        "vel_in": inlet_flow/inlet_area,
                        "res_1": 100,
                        "res_2": 100}

        num_time_steps = 100; time_step_size = 0.2/num_time_steps

        dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}"
        results_dir = f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{geo_name}/"

        if os.path.exists(results_dir) == False:
            os.mkdir(results_dir)


        if os.path.exists(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}"):
            os.system(f"rm -r /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}")
        os.system(f"mkdir /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}")

        num_time_steps = 100
        write_svpre_unsteady(anatomy, geo_name, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps, time_step_size)
        write_unsteady_inp(anatomy, geo_name, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps, time_step_size)
        write_unsteady_flow(anatomy, geo_name, flow_params["flow_amp"], inlet_cap_number, num_time_steps, time_step_size)
        #print("done writing files")
        f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}/numstart.dat", "w"); f.write("0"); f.close()

        print(f"Writing job for {geo}")
        write_geo_job_unsteady(anatomy, geo, flow_name = "unsteady", flow_index = "unsteady", num_cores = num_cores, num_time_steps = num_time_steps)
        print(f"Running job for {geo}")
        os.system(f"sbatch /scratch/users/nrubio/job_scripts/{geo}_unsteady.sh")

    except:
        continue
