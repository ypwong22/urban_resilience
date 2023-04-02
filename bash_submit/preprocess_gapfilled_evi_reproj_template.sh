#!/bin/bash
#SBATCH -A CLI146
#SBATCH -J preprocess_gapfilled_evi_reproj_REPLACE1_REPLACE2
#SBATCH -o preprocess_gapfilled_evi_reproj_REPLACE1_REPLACE2.j%j
#SBATCH -o preprocess_gapfilled_evi_reproj_REPLACE1_REPLACE2.j%j
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --export=ALL

cd ~/Git/sm_eco
ipython -i preprocess_gapfilled_evi_reproj_REPLACE1_REPLACE2.py
