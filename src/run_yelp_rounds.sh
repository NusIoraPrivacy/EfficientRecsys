#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
rounds=(1 2 3)

echo "Train base model: MF, dim 64"
python3 run_train_central.py --model MF --dataset yelp --lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 64 --regularization

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: MF, dim 64, round $rnd"
    run_infer.py --model MF --dataset yelp --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64
done

echo "Train base model: NCF, dim 16"
python3 run_train_central.py --model NCF --dataset yelp --lr 0.001 --n_factors 10

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: NCF, dim 16, round $rnd"
    python3 run_infer.py --model NCF --dataset yelp --d_lr 0.01 --d_dim 5 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 10
done

echo "Train base model: FM, dim 64"
python3 run_train_central.py --model FM --dataset yelp --lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 64 --regularization

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: FM, dim 64, round $rnd"
    python3 run_infer.py --model FM --dataset yelp --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64
done

echo "Train base model: DeepFM, dim 64"
python3 run_train_central.py --model DeepFM --dataset yelp --lr 0.01 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization

for rnd in ${rounds[@]}
do
    echo "Train Denoise model: DeepFM, dim 64, round $rnd"
    python3 run_infer.py --model DeepFM --dataset yelp --d_lr 0.01 --d_dim 8 --l2_reg_u 0.0001 --l2_reg_i 0.0001 --n_factors 64
done