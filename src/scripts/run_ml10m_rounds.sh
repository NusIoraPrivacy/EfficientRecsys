#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
rounds=(1 2 3)

echo "Train base model: MF, dim 64"
python3 run_train_central.py --model MF --dataset ml-10m --lr 0.01 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: MF, dim 64, round $rnd"
    run_infer.py --model MF --dataset ml-10m --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64
done

echo "Train base model: NCF, dim 24"
python3 run_train_central.py --model NCF --dataset ml-10m --lr 0.001 --n_factors 12

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: NCF, dim 24, round $rnd"
    python3 run_infer.py --model NCF --dataset ml-10m --d_lr 0.01 --d_dim 6 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 12
done

echo "Train base model: FM, dim 64"
python3 run_train_central.py --model FM --dataset ml-10m --lr 0.005 --n_factors 64

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: FM, dim 64, round $rnd"
    python3 run_infer.py --model FM --dataset ml-10m --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64
done

echo "Train base model: DeepFM, dim 64"
python3 run_train_central.py --model DeepFM --dataset ml-10m --lr 0.005 --n_factors 64

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: DeepFM, dim 64, round $rnd"
    python3 run_infer.py --model DeepFM --dataset ml-10m --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64
done