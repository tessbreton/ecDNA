#!/bin/sh
#
#SBATCH --account=iicd
#SBATCH --job-name=abctest
#SBATCH -c 32
#SBATCH -t 0-01:00
#SBATCH --mem-per-cpu=5gb

module load anaconda

python abctesting.py