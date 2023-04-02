#!/bin/bash
#SBATCH --mem=32G
#SBATCH -A CLI146
#SBATCH -o summary_uhi_perpixel_REPLACE.j%j
#SBATCH -e summary_uhi_perpixel_REPLACE.j%j
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH --export=ALL


cd ~/Git/sm_eco
python -u summary_uhi_perpixel_REPLACE.py
