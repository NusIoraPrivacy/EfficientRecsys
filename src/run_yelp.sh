#!/bin/bash
# epsilons=(10 8 5 3)
# for eps in ${epsilons[@]}
# do
#     echo "Test denoise model for epsilon $eps"
#     python3 run_infer.py --epsilon=$eps --model MF --dataset yelp --d_lr 0.01 --l2_reg_u 0.01 --l2_reg_i 0.01 --n_factors 64 --d_dim 8
# done

epsilons=(1 50)
for eps in ${epsilons[@]}
do
    echo "Test denoise model for epsilon $eps"
    python3 run_infer.py --epsilon=$eps --model MF --dataset ml-1m --d_lr 0.01 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --d_dim 8
done