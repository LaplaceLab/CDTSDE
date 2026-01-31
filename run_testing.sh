#!/bin/bash -l
if [ -z "$1" ]; then
  echo "Usage: $0 <checkpoint_path> [output_dir]"
  exit 1
fi

ckpt_path="$1"
output_dir="${2:-results/pathology}"

python inference.py --model_config configs/model/cldm_v21_dynamic.yaml \
                   --ckpt "$ckpt_path" \
                   --data_config configs/dataset/pathology/paired_test.yaml \
                   --output "$output_dir" \
                   --batch_size 1 --steps 50
python evaluation.py --results_dir "$output_dir"
