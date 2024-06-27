#!/bin/sh
#
#SBATCH --account=iicd
#SBATCH --job-name=simulations
#SBATCH -c 32
#SBATCH -t 0-01:00
#SBATCH --mem-per-cpu=5gb

module load anaconda

python runsimulations.py --expname runs/P-5 --sample selection --start -5