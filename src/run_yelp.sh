#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

echo "Train base model: MF, dim 512"
python3 run_train_central.py --model MF --dataset yelp --lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 512 --regularization

echo "Train Denoise model: MF, dim 512"
python3 run_infer.py --model MF --dataset yelp --d_lr 0.01 --d_dim 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 512

echo "Train base model: NCF, dim 128"
python3 run_train_central.py --model NCF --dataset yelp --lr 0.001 --n_factors 64

echo "Train Denoise model: NCF, dim 128"
python3 run_infer.py --model NCF --dataset yelp --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64

echo "Train base model: FM, dim 512"
python3 run_train_central.py --model FM --dataset yelp --lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 512 --regularization

echo "Train Denoise model: FM, dim 512"
python3 run_infer.py --model FM --dataset yelp --d_lr 0.01 --d_dim 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 512

echo "Train base model: DeepFM, dim 512"
python3 run_train_central.py --model DeepFM --dataset yelp --lr 0.01 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 512 --regularization

echo "Train Denoise model: DeepFM, dim 512"
python3 run_infer.py --model DeepFM --dataset yelp --d_lr 0.01 --d_dim 12 --l2_reg_u 0.0001 --l2_reg_i 0.0001 --n_factors 512

# epsilons=(1 50)
# for eps in ${epsilons[@]}
# do
#     echo "Test denoise model for epsilon $eps"
#     python3 run_infer.py --epsilon=$eps --model MF --dataset ml-1m --d_lr 0.01 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --d_dim 8
# done