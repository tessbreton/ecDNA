#!/bin/sh
#
#SBATCH --account=iicd         # Replace ACCOUNT with your group account name
#SBATCH --job-name=synthetictesting     # The job name.
#SBATCH -c 32                   # The number of cpu cores to use
#SBATCH -t 1-00:00                # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb         # The memory the job will use per cpu core

module load anaconda

python synthetictesting.py