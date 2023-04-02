#!/bin/bash
#SBATCH --mem=32G
#SBATCH -A CLI146
#SBATCH -o monthly_fit_examine_REPLACE.j%j
#SBATCH -e monthly_fit_examine_REPLACE.j%j
#SBATCH -N 1
#SBATCH -t 00:59:00
#SBATCH --export=ALL


cd ~/Git/sm_eco
python -u monthly_fit_examine_REPLACE.py
