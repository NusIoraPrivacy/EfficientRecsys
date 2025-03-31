#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
rounds=(1)

for rnd in ${rounds[@]}
do
    # echo "Train base model: NCF, compress: 8intquant, round $rnd"
    # python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 8 --batch_size 100 --dataset amazon --compress 8intquant

    # echo "Train base model: NCF, compress: ternquant, round $rnd"
    # python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 8 --batch_size 100 --dataset amazon --compress ternquant

    # echo "Train base model: NCF, compress: svd, round $rnd"
    # python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 8 --batch_size 100 --dataset amazon --compress svd

    echo "Train base model: NCF, compress: colr, round $rnd"
    python3 run_train_central_colr.py --model NCF --lr 0.0001 --n_factors 8 --batch_size 100 --dataset amazon --compress colr
done