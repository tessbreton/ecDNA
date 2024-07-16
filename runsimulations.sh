#!/bin/sh
#
#SBATCH --account=iicd
#SBATCH --job-name=simulations
#SBATCH -c 32
#SBATCH -t 0-00:05
#SBATCH --mem-per-cpu=5gb

module load anaconda

python runsimulations.py --expname experiments/P4P15/runs --sample selection