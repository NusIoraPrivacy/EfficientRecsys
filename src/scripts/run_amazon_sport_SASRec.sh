#!/bin/bash
# export CUDA_VISIBLE_DEVICES=2
rounds=(1 2 3)
compresses=(none ternquant 8intquant colr svd)

for rnd in ${rounds[@]}
do
    for cpr in ${compresses[@]}
    do  
        if [ $cpr = "svd" ]; then
            echo "Train base model: SASRec, compress: $cpr, round $rnd"
            python3 -m SASRec.main --compress $cpr --dataset amazon-sport --device cuda:2 --num_epochs 50 --early_stop 5
        else
            echo "Train base model: SASRec, compress: $cpr, round $rnd"
            python3 -m SASRec.main --compress $cpr --dataset amazon-sport --device cuda:2 --num_epochs 100 --early_stop 10
        fi
    done
done