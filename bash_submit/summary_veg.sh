#!/bin/bash
#SBATCH --mem=128G
#SBATCH -A CLI146
#SBATCH -o summary_veg.j%j
#SBATCH -e summary_veg.j%j
#SBATCH -N 2
#SBATCH -t 10:00:00
#SBATCH --export=ALL

cd ~/Git/sm_eco
python -u summary_veg.py
