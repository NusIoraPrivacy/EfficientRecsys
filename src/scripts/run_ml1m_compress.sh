#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
rounds=(1)

for rnd in ${rounds[@]}
do
    echo "Train base model: MF, compress: 8intquant, round $rnd"
    python3 run_train_central.py --model MF --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress 8intquant --batch_size 100

    echo "Train base model: MF, compress: ternquant, round $rnd"
    python3 run_train_central.py --model MF --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress ternquant --batch_size 100

    echo "Train base model: MF, compress: svd, round $rnd"
    python3 run_train_central.py --model MF --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress svd --batch_size 100 --rank 10

    echo "Train base model: MF, compress: colr, round $rnd"
    python3 run_train_central_colr.py --model MF --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress colr --batch_size 100 --rank 10

    echo "Train base model: NCF, compress: 8intquant, round $rnd"
    python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 8 --compress 8intquant --batch_size 100

    echo "Train base model: NCF, compress: ternquant, round $rnd"
    python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 8 --compress ternquant --batch_size 100

    echo "Train base model: NCF, compress: svd, round $rnd"
    python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 8 --compress svd --batch_size 100 --rank 8

    echo "Train base model: NCF, compress: colr, round $rnd"
    python3 run_train_central_colr.py --model NCF --lr 0.0001 --n_factors 8 --compress colr --batch_size 100 --rank 8

    echo "Train base model: FM, compress: 8intquant, round $rnd"
    python3 run_train_central.py --model FM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress 8intquant --batch_size 100

    echo "Train base model: FM, compress: ternquant, round $rnd"
    python3 run_train_central.py --model FM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress ternquant --batch_size 100

    echo "Train base model: FM, compress: svd, round $rnd"
    python3 run_train_central.py --model FM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress svd --batch_size 100 --rank 10

    echo "Train base model: FM, compress: colr, round $rnd"
    python3 run_train_central_colr.py --model FM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress colr --batch_size 100 --rank 10

    echo "Train base model: DeepFM, compress: 8intquant, round $rnd"
    python3 run_train_central.py --model DeepFM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress 8intquant --batch_size 100

    echo "Train base model: DeepFM, compress: ternquant, round $rnd"
    python3 run_train_central.py --model DeepFM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress ternquant --batch_size 100

    echo "Train base model: DeepFM, compress: svd, round $rnd"
    python3 run_train_central.py --model DeepFM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress svd --batch_size 100 --rank 10

    echo "Train base model: DeepFM, compress: colr, round $rnd"
    python3 run_train_central_colr.py --model DeepFM --lr 0.025 --l2_reg_u 0.001 --l2_reg_i 0.001 --n_factors 64 --regularization --compress colr --batch_size 100 --rank 10
done