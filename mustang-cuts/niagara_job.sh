#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:00
#SBATCH --job-name=neatnik_mustang
#SBATCH --output=/scratch/r/rbond/fzhs/neatnik_mustang_%j.txt
#SBATCH --mail-type=FAIL
 
cd $SLURM_SUBMIT_DIR

module load python/3.9.8
module load cmake/3.17.3
module load intel/2019u4
module load intelmpi/2019u4
 
mpirun python3 mustang_cuts.py
