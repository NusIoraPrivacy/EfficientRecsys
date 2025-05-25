#!/bin/bash
# export CUDA_VISIBLE_DEVICES=2
shopt -s extglob
rounds=(1 2)
compresses=(none ternquant 8intquant colr svd)
targets=(svd ternquant none)

for rnd in ${rounds[@]}
do
    for cpr in ${compresses[@]}
    do  
        is_target=false
        for tgt in "${targets[@]}"; do
            if [[ "$tgt" == "$cpr" ]]; then
                is_target=true
                break
            fi
        done

        if $is_target && (($rnd < 2)); then
            echo "Train base model: Caser, compress: $cpr, round $rnd"
            python3 -m Caser.main --compress $cpr --dataset amazon-sport --device cuda:1 --num_epochs 50 --early_stop 5
        else
            echo "Train base model: Caser, compress: $cpr, round $rnd"
            python3 -m Caser.main --compress $cpr --dataset amazon-sport --device cuda:1 --num_epochs 100 --early_stop 10
        fi
    done
done