#!/bin/bash
# export CUDA_VISIBLE_DEVICES=2
rounds=(1 2)
compresses=(none ternquant 8intquant colr svd)

for rnd in ${rounds[@]}
do
    for cpr in ${compresses[@]}
    do  
        if [ $cpr = "svd" ]; then
            echo "Train base model: Caser, compress: $cpr, round $rnd"
            python3 -m Caser.main --compress $cpr --dataset amazon-beauty --device cuda:2 --num_epochs 100 --early_stop 10
        else
            echo "Train base model: Caser, compress: $cpr, round $rnd"
            python3 -m Caser.main --compress $cpr --dataset amazon-beauty --device cuda:2 --num_epochs 200 --early_stop 20
        fi
    done
done