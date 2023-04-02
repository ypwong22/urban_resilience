#!/bin/bash
#SBATCH -p batch
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -t 18:00:00
#SBATCH --export=ALL
#SBATCH -A CLI146
#SBATCH -J convert_prcp_spi_REPLACE

##SBATCH -A ccsi
##SBATCH -o convert_prcp_spi_REPLACE.j%j
##SBATCH -e convert_prcp_spi_REPLACE.j%j

cd ~/Git/sm_eco
ipython -i convert_prcp_spi_REPLACE.py