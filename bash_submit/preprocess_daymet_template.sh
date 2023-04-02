#!/bin/bash
#SBATCH -A ccsi
#SBATCH -p batch
#SBATCH --mem=32G
#SBATCH -e preprocess_daymet_REPLACE1_REPLACE2.j%j
#SBATCH -o preprocess_daymet_REPLACE1_REPLACE2.j%j
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH --export=ALL

###SBATCH -A CLI146
###SBATCH -J preprocess_daymet_REPLACE1_REPLACE2

cd ~/Git/sm_eco
ipython -i preprocess_daymet_REPLACE1_REPLACE2.py
