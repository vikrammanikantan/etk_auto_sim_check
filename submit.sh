#!/bin/sh
#SBATCH --job-name=sim_check
#SBATCH -o sim_check.out
#SBATCH -e sim_check.err
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --mail-type=all
#SBATCH --mail-user=vik@arizona.edu

module unload impi
module load hdf5/1.14.4

export MPI_DISABLE=1
export HDF5_USE_FILE_LOCKING='FALSE'
export OMPI_MCA_plm=isolated
export OMPI_MCA_btl=^openib

python3 send_email.py DISK_A30W_new 225
