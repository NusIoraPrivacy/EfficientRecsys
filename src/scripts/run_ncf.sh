#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
rounds=(1)

for rnd in ${rounds[@]}
do
    # echo "Train base model: NCF, dataset: ml-1m, dim 24, round $rnd"
    # python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 12 --dataset ml-1m

    # echo "Train base model: NCF, dataset: ml-100k, dim 24, round $rnd"
    # python3 run_train_central.py --model NCF --dataset ml-100k --lr 0.001 --n_factors 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --regularization

    echo "Train base model: NCF, dataset: yelp, dim 24, round $rnd"
    python3 run_train_central.py --model NCF --dataset yelp --lr 0.001 --n_factors 12

    echo "Train base model: NCF, dataset: yelp, dim 24, compress: 8intquant, round $rnd"
    python3 run_train_central.py --model NCF --dataset yelp --lr 0.001 --n_factors 12 --compress 8intquant --batch_size 100

    echo "Train base model: NCF, dataset: yelp, dim 24, compress: ternquant, round $rnd"
    python3 run_train_central.py --model NCF --dataset yelp --lr 0.001 --n_factors 12 --compress ternquant --batch_size 100

    echo "Train base model: NCF, dataset: yelp, dim 24, compress: svd, round $rnd"
    python3 run_train_central.py --model NCF --dataset yelp --lr 0.001 --n_factors 12 --compress svd --batch_size 100 --rank 4

    echo "Train base model: NCF, dataset: yelp, dim 24, compress: colr, round $rnd"
    python3 run_train_central_colr.py --model NCF --dataset yelp --lr 0.001 --n_factors 12 --compress colr --batch_size 100 --rank 4

    # echo "Train base model: NCF, dataset: ml-1m, compress: 8intquant, round $rnd"
    # python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 12 --compress 8intquant --batch_size 100 --dataset ml-1m

    # echo "Train base model: NCF, dataset: ml-1m, compress: ternquant, round $rnd"
    # python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 12 --compress ternquant --batch_size 100 --dataset ml-1m

    # echo "Train base model: NCF, dataset: ml-1m, compress: svd, round $rnd"
    # python3 run_train_central.py --model NCF --lr 0.0001 --n_factors 12 --compress svd --batch_size 100 --rank 8 --dataset ml-1m

    # echo "Train base model: NCF, dataset: ml-1m, compress: colr, round $rnd"
    # python3 run_train_central_colr.py --model NCF --lr 0.0001 --n_factors 12 --compress colr --batch_size 100 --rank 8 --dataset ml-1m

    # echo "Train base model: NCF, dataset: ml-100k, compress: 8intquant, round $rnd"
    # python3 run_train_central.py --model NCF --dataset ml-100k --lr 0.001 --n_factors 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --regularization --compress 8intquant --batch_size 100

    # echo "Train base model: NCF, dataset: ml-100k, compress: ternquant, round $rnd"
    # python3 run_train_central.py --model NCF --dataset ml-100k --lr 0.001 --n_factors 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --regularization --compress ternquant --batch_size 100

    # echo "Train base model: NCF, dataset: ml-100k, compress: svd, round $rnd"
    # python3 run_train_central.py --model NCF --dataset ml-100k --lr 0.001 --n_factors 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --regularization --compress svd --batch_size 100 --rank 10

    # echo "Train base model: NCF, dataset: ml-100k, compress: colr, round $rnd"
    # python3 run_train_central_colr.py --model NCF --dataset ml-100k --lr 0.001 --n_factors 12 --l2_reg_u 0.001 --l2_reg_i 0.001 --regularization --compress colr --batch_size 100 --rank 10
done