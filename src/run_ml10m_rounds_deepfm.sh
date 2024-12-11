#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
rounds=(1 2 3)

echo "Train base model: DeepFM, dim 64"
python3 run_train_central.py --model DeepFM --dataset ml-10m --lr 0.005 --n_factors 64

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: DeepFM, dim 64, round $rnd"
    python3 run_infer.py --model DeepFM --dataset ml-10m --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64
done