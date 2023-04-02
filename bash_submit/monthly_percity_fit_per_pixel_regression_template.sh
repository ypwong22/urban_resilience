#!/bin/bash
#SBATCH -A cli146
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -e monthly_percity_fit_per_pixel_regression_REPLACE0_REPLACE2_REPLACE3_REPLACE4_REPLACE5.j%j
#SBATCH -o monthly_percity_fit_per_pixel_regression_REPLACE0_REPLACE2_REPLACE3_REPLACE4_REPLACE5.j%j
#SBATCH --export=ALL

cd ~/Git/sm_eco
python -u monthly_percity_fit_per_pixel_regression_REPLACE0_REPLACE2_REPLACE3_REPLACE4_REPLACE5.py
