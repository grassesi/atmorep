#!/bin/bash
echo "load hpc-modules"

ml --force purge
ml use $OTHERSTAGES
ml Stages/2024

ml GCC/12.3.0
ml GCCcore/.12.3.0

ml OpenMPI/4.1.5
ml ecCodes/2.31.0 # needs openmpi (serial version exists)

ml Python/3.11.3

# serial versions get used
# ml HDF5/1.14.2
# ml netCDF/4.9.2

# python modules
ml PyTorch/2.1.2
ml dask/2023.9.2
ml netcdf4-python/1.6.4-serial
ml matplotlib/3.7.2
ml SciPy-bundle/2023.07
ml xarray/2023.8.0
ml Cartopy/0.22.0