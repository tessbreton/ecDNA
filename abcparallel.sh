#!/bin/sh
#
#SBATCH --account=iicd
#SBATCH --job-name=ecDNA
#SBATCH -c 32
#SBATCH -t 0-00:45
#SBATCH --mem-per-cpu=5gb

module load anaconda

#Command to execute Python program
python abcparallel.py
 
#End of script