#!/bin/bash
compresses=(none ternquant 8intquant colr svd)
rounds=(1 2 3)

for rnd in ${rounds[@]}
do
    for cpr in ${compresses[@]}
    do  
        echo "Train base model: Roberta, compress: $cpr, round $rnd"
        python3 -m lm.test_utility --compress $cpr --dataset sst2 --device cuda:0 --epochs 10
    done
done