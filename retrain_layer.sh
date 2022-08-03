#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch_size 32 --transformer --optim_config 