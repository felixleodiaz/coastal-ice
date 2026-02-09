#!/bin/bash

SBATCH --job-name=coastal_ice            # job name
SBATCH --partition=hpc                   # partition or queue name
SBATCH  -N1                              # num of nodes (I want one entire node)
SBATCH --time=10:00:00                   # max runtime
SBATCH --output=job_output_%j.out        # job output
SBATCH --error=job_output_%j.err         # job error output
SBATCH --mail-type=START,END,FAIL        # send email at start and end of job
SBATCH --mail-user=fld1@williams.edu     # email address for notifications

# load python interpreter
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lgb-env

# run scripts
python machine_learning.py