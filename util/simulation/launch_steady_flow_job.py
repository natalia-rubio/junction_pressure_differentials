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

# def write_geo_job(anatomy, geo_name, flow_name, flow_index):
#
#     geo_job_script = f"#!/bin/bash\n\
# # Name of your job\n\
# #SBATCH --job-name={geo_name}_flow_sweep\n\
# # Name of partition\n\
# #SBATCH --partition=amarsden\n\
# #SBATCH --output=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.o%j\n\
# #SBATCH --error=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.e%j\n\
# # The walltime you require for your simulation\n\
# #SBATCH --time=00:60:00\n\
# # Amount of memory you require per node. The default is 4000 MB (or 4 GB) per node\n\
# #SBATCH --mem=10000\n\
# #SBATCH --nodes=1\n\
# #SBATCH --tasks-per-node=24\n\
# # Load Modules\n\
# module purge\n\
# module load openmpi\n\
# module load openblas\n\
# module load system\n\
# module load x11\n\
# module load mesa\n\
# module load gcc\n\
# module load valgrind\n\
# module load python/3.9.0\n\
# module load py-numpy/1.20.3_py39\n\
# module load py-scipy/1.6.3_py39\n\
# module load gcc/10.1.0\n\
# # Name of the executable you want to run\n\
# source /home/users/nrubio/junctionenv/bin/activate\n\
# /home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpre.exe /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_job.svpre\n\
# echo 'Done with svPre.'\n\
# conv=false\n\
# conv_attempts=1\n\
# indir='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}/24-procs_case'\n\
# outdir='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}'\n\
# while (( !conv && conv_attempts < 20 ))\n\
# do\n\
#     echo 'Launching Simulation'\n\
#     cd /scratch/users/nrubio/synthetic_junctions/Aorta/{geo}/{flow_name}\n\
#     mpirun -n 24 /home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svsolver-openmpi.exe {flow_name}_solver.inp\n\
#     echo 'Simulation completed'\n\
#     /home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpost.exe -start $(( 100*conv_attempts -100 )) -stop $(( 100*conv_attempts )) -incr 100 -vtkcombo -indir $indir -outdir $outdir -vtu solution_flow_{flow_index}.vtu\n\
#     python3 /home/users/nrubio/SV_scripts/for_sherlock/check_convergence.py {geo_name} {flow_index} {anatomy}\n\
#     conv_attempts=$(( conv_attempts+1 ))\n\
#     if [ $? -eq 1 ];\n\
#     then\n\
#         conv=true\n\
#     fi\n\
# rmkk -r $outdir"
#     f = open(f"/scratch/users/nrubio/job_scripts/{geo}_flow_{i}.sh", "w")
#     f.write(geo_job_script)
#     f.close()
#     return
def write_geo_job(anatomy, geo_name, flow_name, flow_index, num_cores, num_time_steps):

    geo_job_script = f"#!/bin/bash\n\
# Name of your job\n\
#SBATCH --job-name={geo_name}_flow_sweep\n\
# Name of partition\n\
#SBATCH --partition=amarsden\n\
#SBATCH --output=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.o%j\n\
#SBATCH --error=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.e%j\n\
# The walltime you require for your simulation\n\
#SBATCH --time=00:90:00\n\
# Amount of memory you require per node. The default is 4000 MB (or 4 GB) per node\n\
#SBATCH --mem=100000\n\
#SBATCH --nodes=2\n\
#SBATCH --tasks-per-node=24\n\
# Load Modules\n\
module purge\n\
module load openmpi\n\
module load openblas\n\
module load system\n\
module load x11\n\
module load mesa\n\
module load viz\n\
module load gcc\n\
module load valgrind\n\
module load python/3.9.0\n\
module load py-numpy/1.20.3_py39\n\
module load py-scipy/1.6.3_py39\n\
module load py-scikit-learn/1.0.2_py39\n\
module load gcc/10.1.0\n\
# Name of the executable you want to run\n\
source /home/users/nrubio/junctionenv/bin/activate\n\
/home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpre.exe /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_job.svpre\n\
echo 'Done with svPre.'\n\
conv=false\n\
conv_attempts=1\n\
indir='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}/{num_cores}-procs_case'\n\
outdir='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}'\n\
echo 'Launching Simulation'\n\
cd /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo}/{flow_name}\n\
mpirun -n {num_cores} /home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svsolver-openmpi.exe {flow_name}_solver.inp\n\
echo 'Simulation completed'\n\
/home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpost.exe -start {num_time_steps-100} -stop {num_time_steps} -incr 100 -vtkcombo -indir $indir -outdir $outdir -vtu solution_flow_{flow_index}.vtu > /dev/null\n\
python3 /home/users/nrubio/SV_scripts/for_sherlock/check_convergence.py {geo_name} {flow_index} {anatomy} {num_time_steps}\n\
kkrm -r $outdir"
    f = open(f"/scratch/users/nrubio/job_scripts/{geo}_flow_{i}.sh", "w")
    f.write(geo_job_script)
    f.close()
    return


anatomy = sys.argv[1]; num_time_steps = sys.argv[2]; num_cores = sys.argv[3]; num_geos = sys.argv[4]

num_geos = 150; num_launched = 0
print(f"Launching {num_geos} steady flow sweeps.")
anatomy = "mynard_vary_mesh_sphere"#"mynard_over9"#"Aorta_u_40-60_over15"
dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}"
geos = os.listdir(dir); geos.sort(); geo_ind = 0;#

while num_launched < num_geos:

    geo = geos[geo_ind]; geo_name = geo; print(geo_name)
    geo_ind += 1
    if geo[0] != "A":
        print("Not an aorta.");# continue

    already_done = True
    for j in range(4):
        if not os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{geo}/flow_{j}_red_sol"):
            already_done = False
    if already_done: print("4 steady simulations already complete"); continue

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

    for i, inlet_flow_fac in enumerate([0.25, 0.5, 0.75, 1]):

        try:
            flow_index = i; flow_name = f"flow_{flow_index}"
            if i == 0 or i == 2:
                continue
            if os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{geo_name}/flow_{i}_red_sol"):
                print(f"Simulation already complete for flow {flow_index}")
                continue

            flow_params = {"flow_amp": inlet_flow*inlet_flow_fac,
                            "vel_in": inlet_flow*inlet_flow_fac/inlet_area,
                            "res_1": 100,
                            "res_2": 100}

            print("Starting run_simulation function.");
            dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}"
            results_dir = f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{geo_name}/"



            if os.path.exists(results_dir) == False:
                os.mkdir(results_dir)


            if os.path.exists(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}"):
                os.system(f"rm -r /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}")
            os.system(f"mkdir /scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}")

            time_step_size = 0.001
            write_svpre_steady(anatomy, geo_name, flow_index, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps, time_step_size)
            write_steady_inp(anatomy, geo_name, flow_index, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps, time_step_size)
            write_steady_flow(anatomy, geo_name, flow_index, flow_params["flow_amp"], inlet_cap_number, num_time_steps, time_step_size)
            print("done writing files")
            f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}/numstart.dat", "w"); f.write("0"); f.close()

            print(f"Started job for {geo} flow {flow_index}")
            write_geo_job(anatomy, geo, flow_name = flow_name, flow_index = flow_index, num_cores, num_time_steps)
            os.system(f"sbatch /scratch/users/nrubio/job_scripts/{geo[0]}_f{i}.sh")

        except:
            continue
    num_launched +=1
