#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=0:30:00
#SBATCH --job-name=neatnik_mustang
#SBATCH --output=/scratch/r/rbond/fzhs/neatnik_mustang_%j.txt
#SBATCH --mail-type=FAIL
 
cd $SLURM_SUBMIT_DIR

module load gcc/8.3.0
module load fftw/3.3.8
module load openmpi/3.1.3
module load cmake/3.17.3
module load python/3.9.8
 
mpirun python3 mustang_cuts.py
