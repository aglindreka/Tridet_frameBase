#!/bin/bash

#echo "start training"
#CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/thumos_i3d_mpiigi.yaml --output mpiigi
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/thumos_i3d_mpiigi.yaml ckpt_mpiigi/thumos_i3d_mpiigi_mpiigi/epoch_039.pth.tar
