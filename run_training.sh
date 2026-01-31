#!/bin/bash -l
python train.py --config "${1:-configs/train_cldm_paired_dynamic_pathology.yaml}"
