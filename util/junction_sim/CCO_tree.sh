#!/bin/bash
# Name of your job
#SBATCH --job-name={geo_name}_flow_sweep
# Name of partition
#SBATCH --partition=amarsden
#SBATCH --output=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.o%j
#SBATCH --error=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.e%j
# The walltime you require for your simulation
#SBATCH --time=10:00:00
# Amount of memory you require per node. The default is 4000 MB (or 4 GB) per node
#SBATCH --mem=50000
#SBATCH --nodes=4
#SBATCH --tasks-per-node=24
# Load Modules
module purge
module load openmpi
module load openblas
module load system
module load x11
module load mesa
module load viz
module load gcc
module load valgrind
module load python/3.9.0
module load py-numpy/1.20.3_py39
module load py-scipy/1.6.3_py39
module load py-scikit-learn/1.0.2_py39
module load gcc/10.1.0

source /home/users/nrubio/junctionenv/bin/activate
mpirun -n 96 /home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svsolver-openmpi.exe /scratch/users/nrubio/synthetic_junctions/CCO/simulation_data9/solver.inp