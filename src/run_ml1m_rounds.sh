#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
rounds=(1 2 3)

for rnd in ${rounds[@]}
do
    echo "Train base model: MF, dim 64, round $rnd"
    python3 run_train_central.py --model MF --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization

    echo "Train Denoise model: MF, dim 64, round $rnd"
    python3 run_infer.py --model MF --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64

    echo "Train base model: NCF, dim 16, round $rnd"
    python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 8

    echo "Train Denoise model: NCF, dim 16, round $rnd"
    python3 run_infer.py --model NCF --d_lr 0.01 --d_dim 5 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 8

    echo "Train base model: FM, dim 64, round $rnd"
    python3 run_train_central.py --model FM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization

    echo "Train Denoise model: FM, dim 64, round $rnd"
    python3 run_infer.py --model FM --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64

    echo "Train base model: DeepFM, dim 64, round $rnd"
    python3 run_train_central.py --model DeepFM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization

    echo "Train Denoise model: DeepFM, dim 64, round $rnd"
    python3 run_infer.py --model DeepFM --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64
done