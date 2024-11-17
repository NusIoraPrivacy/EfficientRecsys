#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
epsilons=(0.1 1 10)

# echo "Train base model: MF, dim 512"
# python3 run_train_central.py --model MF --dataset yelp --lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 512 --regularization

# echo "Train Denoise model: MF, dim 512"
# python3 run_infer.py --model MF --dataset yelp --d_lr 0.01 --d_dim 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 512

# echo "Train base model: NCF, dim 128"
# python3 run_train_central.py --model NCF --dataset yelp --lr 0.001 --n_factors 64

# echo "Train Denoise model: NCF, dim 128"
# python3 run_infer.py --model NCF --dataset yelp --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64

# echo "Train base model: FM, dim 512"
# python3 run_train_central.py --model FM --dataset yelp --lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 512 --regularization

# echo "Train Denoise model: FM, dim 512"
# python3 run_infer.py --model FM --dataset yelp --d_lr 0.01 --d_dim 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 512

# echo "Train base model: DeepFM, dim 512"
# python3 run_train_central.py --model DeepFM --dataset yelp --lr 0.01 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 512 --regularization

# echo "Train Denoise model: DeepFM, dim 512"
# python3 run_infer.py --model DeepFM --dataset yelp --d_lr 0.01 --d_dim 12 --l2_reg_u 0.0001 --l2_reg_i 0.0001 --n_factors 512

echo "Train base model: MF, dim 64"
python3 run_train_central.py --model MF --dataset yelp --lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 64 --regularization

for eps in ${epsilons[@]}
do
    echo "Train Denoise model: MF, dim 64, epsilon $eps"
    run_infer.py --model MF --dataset yelp --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --epsilon=$eps
done

echo "Train base model: NCF, dim 20"
python3 run_train_central.py --model NCF --dataset yelp --lr 0.001 --n_factors 10

for eps in ${epsilons[@]}
do
    echo "Train Denoise model: NCF, dim 20, epsilon $eps"
    python3 run_infer.py --model NCF --dataset yelp --d_lr 0.01 --d_dim 5 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 10 --epsilon=$eps
done

echo "Train base model: FM, dim 64"
python3 run_train_central.py --model FM --dataset yelp --lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 64 --regularization

for eps in ${epsilons[@]}
do
    echo "Train Denoise model: FM, dim 64, epsilon $eps"
    python3 run_infer.py --model FM --dataset yelp --d_lr 0.01 --d_dim 8 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --epsilon=$eps
done

echo "Train base model: DeepFM, dim 64"
python3 run_train_central.py --model DeepFM --dataset yelp --lr 0.01 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization

for eps in ${epsilons[@]}
do
    echo "Train Denoise model: DeepFM, dim 64, epsilon $eps"
    python3 run_infer.py --model DeepFM --dataset yelp --d_lr 0.01 --d_dim 8 --l2_reg_u 0.0001 --l2_reg_i 0.0001 --n_factors 64 --epsilon=$eps
done