#!/bin/bash
#SBATCH --mem=128G
#SBATCH -A CLI146
#SBATCH -o monthly_fit_metrics.j%j
#SBATCH -e monthly_fit_metrics.j%j
#SBATCH -N 2
#SBATCH -t 10:00:00
#SBATCH --export=ALL


cd ~/Git/sm_eco
python -u monthly_fit_metrics.py
