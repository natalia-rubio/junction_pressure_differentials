import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.simulation.get_avg_sol import *
from util.simulation.local.local_write_solver_files import *
from util.tools.junction_proc import *

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

def write_geo_job(anatomy, geo_name, flow_name, flow_index):

    geo_job_script = f"#!/bin/bash\n\
# Name of your job\n\
#SBATCH --job-name={geo_name}_flow_sweep\n\
# Name of partition\n\
#SBATCH --partition=amarsden\n\
#SBATCH --output=data/job_scripts/{geo_name}_{flow_name}.o%j\n\
#SBATCH --error=data/job_scripts/{geo_name}_{flow_name}.e%j\n\
# The walltime you require for your simulation\n\
#SBATCH --time=00:30:00\n\
# Amount of memory you require per node. The default is 4000 MB (or 4 GB) per node\n\
#SBATCH --mem=10000\n\
#SBATCH --nodes=1\n\
#SBATCH --tasks-per-node=24\n\
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
/home/nrubio/work/repos/svSolver/build/svSolver-build/bin/svpre data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_job.svpre\n\
echo 'Done with svPre.'\n\
conv=false\n\
conv_attempts=1\n\
indir='data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}/24-procs_case'\n\
outdir='data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}'\n\
echo 'Launching Simulation'\n\
cd /home/nrubio/Desktop/data/synthetic_junctions/{anatomy}/{geo}/{flow_name}\n\
mpirun -n 24 /home/nrubio/work/repos/svSolver/build/svSolver-build/bin/svsolver {flow_name}_solver.inp\n\
echo 'Simulation completed'\n\
/home/nrubio/work/repos/svSolver/build/svSolver-build/bin/svpost -start 900 -stop 1000 -incr 100 -vtkcombo -indir $indir -outdir $outdir -vtu solution_flow_{flow_index}.vtu > /dev/null\n\
python3 data/SV_scripts/for_sherlock/check_convergence.py {geo_name} {flow_index} {anatomy}\n\
rmkk -r $outdir"
    f = open(f"data/job_scripts/{geo}_flow_{i}.sh", "w")
    f.write(geo_job_script)
    f.close()
    return

if __name__ == "__main__":

    num_geos = 1; num_launched = 0
    print(f"Launching {num_geos} steady flow sweeps.")
    anatomy = "mynard"
    dir = f"data/synthetic_junctions/{anatomy}"
    geos = os.listdir(dir); geos.sort(); geo_ind = 0; #geos = ["Aorta_1_coarse"]

    while num_launched < num_geos:

        geo = geos[geo_ind]; geo_name = geo; print(geo_name)
        geo_ind += 1
        if geo[0] != "A":
            print("Not an aorta."); #continue

        if not os.path.exists(f"data/synthetic_junctions_reduced_results/{anatomy}"):
            os.mkdir(f"data/synthetic_junctions_reduced_results/{anatomy}")
        already_done = True
        for j in range(4):
            if not os.path.exists(f"data/synthetic_junctions_reduced_results/{anatomy}/{geo}/flow_{j}_red_sol"):
                already_done = False
        if already_done: print("4 steady simulations already complete"); continue

        try:
            centerline_dir = f"data/synthetic_junctions/{anatomy}/{geo_name}/centerlines/centerline.vtp"
            pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = load_centerline_data(fpath_1d = centerline_dir)
            junction_dict, junc_pt_ids = identify_junctions_synthetic(junction_id, branch_id, pt_id)
        except:
            print(f"Couldn't process centerline.  Removing {geo_name}")
            # os.system(f"rm -r data/synthetic_junctions/{anatomy}/{geo_name}")
            continue

        try:
            params_dict = load_dict(f"data/synthetic_junctions/{anatomy}/{geo_name}/junction_params_dict")
            dir = f"data/synthetic_junctions/{anatomy}"
            inlet_flow = params_dict["inlet_flow"]
            inlet_area = np.pi * params_dict["inlet_radius"]**2

            inlet_cap_number = int(np.load(f"{dir}/{geo_name}/max_area_cap.npy")[0])
            cap_numbers = get_cap_numbers(f"{dir}/{geo_name}/mesh-complete/mesh-surfaces/"); print(cap_numbers)

            if len(cap_numbers) != 3:
                print("Wrong number of caps.  Deleting.")
                os.system(f"rm -r data/synthetic_junctions/{anatomy}/{geo_name}")
        except:
            print(f"Couldn't find parameter dictionary.")
            continue

        for i, inlet_flow_fac in enumerate([0.25,]): #, 0.5, 0.75, 1]):

            #try:
            flow_index = i; flow_name = f"flow_{flow_index}"
            if os.path.exists(f"data/synthetic_junctions_reduced_results/{anatomy}/{geo_name}/flow_{i}_red_sol"):
                print(f"Simulation already complete for flow {flow_index}")
                continue

            flow_params = {"flow_amp": inlet_flow*inlet_flow_fac,
                            "vel_in": inlet_flow*inlet_flow_fac/inlet_area,
                            "res_1": 100,
                            "res_2": 100}

            print("Starting run_simulation function.")
            dir = f"data/synthetic_junctions/{anatomy}/{geo_name}"
            results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{geo_name}/"



            if os.path.exists(results_dir) == False:
                os.mkdir(results_dir)

            #
            # if os.path.exists(f"data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}"):
            #     os.system(f"rm -r data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}")
            # os.system(f"mkdir data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}")

            num_time_steps = 1000
            write_svpre_steady(anatomy, geo_name, flow_index, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps)
            write_steady_inp(anatomy, geo_name, flow_index, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps)
            write_steady_flow(anatomy, geo_name, flow_index, flow_params["flow_amp"], inlet_cap_number, num_time_steps)
            #print("done writing files")
            # f = open(f"data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}/numstart.dat", "w"); f.write("0"); f.close()
            #
            # print("Running svPre.")
            # os.system(f"/home/nrubio/work/repos/svSolver/build/svSolver-build/bin/svpre /home/nrubio/Desktop/junction_pressure_differentials/data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_job.svpre")
            #
            # indir= f'/home/nrubio/Desktop/junction_pressure_differentials/data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}'
            # outdir= f'/home/nrubio/Desktop/junction_pressure_differentials/data/synthetic_junctions/{anatomy}/{geo_name}/{flow_name}'
            #
            # print("Running svSolver.")
            # os.system(f"cd /home/nrubio/Desktop/junction_pressure_differentials/data/synthetic_junctions/{anatomy}/{geo}/{flow_name} && mpirun -n 24 /home/nrubio/work/repos/svSolver/build/svSolver-build/bin/svsolver /home/nrubio/Desktop/junction_pressure_differentials/data/synthetic_junctions/{anatomy}/{geo}/{flow_name}/{flow_name}_solver.inp")
            #
            # print("Running svPost.")
            # os.system(f"/home/nrubio/work/repos/svSolver/build/svSolver-build/bin/svpost -start 900 -stop 1000 -incr 100 -vtkcombo -indir {indir} -outdir {outdir} -vtu solution_flow_{flow_index}.vtu > /dev/null")
            os.system(f"python3 /home/nrubio/Desktop/junction_pressure_differentials/util/simulation/local/local_check_convergence.py {anatomy} {geo_name} {flow_index}")
            # except:
            #     continue
        num_launched +=1
