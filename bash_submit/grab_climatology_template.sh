#!/bin/bash
#SBATCH -A CLI146
#SBATCH -J grab_climatology_REPLACE1_REPLACE2
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH --export=ALL

###PBS -S /bin/bash
###PBS -A ACF-UTK0011
###PBS -l nodes=1:ppn=16,walltime=4:00:00

cd ~/Git/sm_eco
ipython -i grab_climatology_REPLACE1_REPLACE2.py
