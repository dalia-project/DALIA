#!/bin/bash -l
#SBATCH --job-name=dalia_alex
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=00:04:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
###SBATCH --constraint=a100_80
###SBATCH --qos=a100multi
###SBATCH --exclusive
#SBATCH --export=NONE

# Change to examples directory
if [[ "$(basename "$(pwd)")" != "examples" ]]; then
    echo ""
    echo "Error: Not in examples directory"
    echo "   Current directory: $(pwd)"
    echo "   Please run this script from the examples/ directory"
    echo ""
    exit 1
fi

# Set DALIA environment variables for examples  
source ../scripts/alex_fau_utils.sh && alex_load_modules && alex_activate_conda_env && alex_set_perfenv
source ../scripts/dalia_job_utils.sh && dalia_set_perfenv && dalia_print_job_config

# --- How to Run ---
# This run script is designed to run on Alex at NHR@FAU
# It uses SLURM for job scheduling and assumes that the user has a working 
# installation of DALIA and its dependencies.  By default, DALIA will exploit  
# job parallelism in a cascade, first at the function evaluation level,
# then at the precision matrix level, finally at the structured solver level.

# --- Parameters ---
# `--solver_min_p` : The minimum number of Processes(/GPUs) to use for the structured 
#                    solver. The default is 1. The maximum number of processes is
# `--max_iter` : The maximum number of iterations of the minimization.

# --- Brainiac Example ---
srun python ./brainiac/run.py --max_iter 100

# --- Gaussian AR1 Example ---
# srun python ./g_ar1/run.py --max_iter 100

# --- Gaussian Regression Example ---
# srun python ./gr/run.py --max_iter 100

# --- Gaussian Spatial Coregional 2 Models (Small) Example ---
# srun python ./gs_coreg2_small/run.py --max_iter 100

# --- Gaussian Spatial Coregional 3 Models (Small) Example ---
# srun python ./gs_coreg3_small/run.py --max_iter 100

# --- Gaussian Spatial Model (Small) Example ---
# srun python ./gs_small/run.py --max_iter 100

# --- Gaussian Spatio-temporal Coregional 2 Models (Small) Example ---
# srun python ./gst_coreg2_small/run.py --solver_min_p 1 --max_iter 100

# --- Gaussian Spatio-temporal Coregional 3 Models (Small) Example ---
# srun python ./gst_coreg3_small/run.py --solver_min_p 1 --max_iter 100

# --- Gaussian Spatio-temporal Model (Large) Example ---
# srun python ./gst_large/run.py --solver_min_p 1 --max_iter 100

# --- Gaussian Spatio-temporal Model (Medium) Example ---
# srun python ./gst_medium/run.py --solver_min_p 1 --max_iter 100

# --- Gaussian Spatio-temporal Model (Small) Example ---
# srun python ./gst_small/run.py --solver_min_p 1 --max_iter 100

# --- Poisson AR1 Example ---
# srun python ./p_ar1/run.py --max_iter 100

# --- Poisson Regression Example ---
# srun python ./pr/run.py --max_iter 100

# --- Poisson Spatio-temporal Model (Small) Example ---
# srun python ./pst_small/run.py --solver_min_p 1 --max_iter 100