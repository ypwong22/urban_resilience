#!/bin/bash
#SBATCH --mem=64G
#SBATCH -A CLI146
#SBATCH -o monthly_percity_fit.j%j
#SBATCH -e monthly_percity_fit.j%j
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --export=ALL


cd ~/Git/sm_eco
python -u monthly_percity_fit.py
