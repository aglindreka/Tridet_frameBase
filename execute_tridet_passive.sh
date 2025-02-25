#/bin/bash

#OAR -p gpu='YES' and host='nefgpu46.inria.fr'

#OAR -l /gpunum=1, walltime=300

#OAR --name ilsvrc_tfcreator

source activate actionformer
python train.py ./configs/thumos_i3d_mpiigi.yaml --output mpi

