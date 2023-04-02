#!/bin/bash
#SBATCH -A ccsi
#SBATCH -p batch
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -e preprocess_bnu_data_reproj_REPLACE.j%j
#SBATCH -o preprocess_bnu_data_reproj_REPLACE.j%j
#SBATCH --export=ALL

###SBATCH -A CLI146
###SBATCH -J preprocess_bnu_data_reproj_REPLACE

cd ~/Git/sm_eco
ipython -i preprocess_bnu_data_reproj_REPLACE.py
