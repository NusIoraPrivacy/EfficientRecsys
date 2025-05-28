#!/bin/bash
compresses=(none ternquant 8intquant colr svd)
rounds=(1 2 3)
k_ratios=(0.2 0.25 0.3)

# for rnd in ${rounds[@]}
# do
#     for cpr in ${compresses[@]}
#     do  
#         echo "Train base model: Roberta, compress: $cpr, round $rnd"
#         python3 -m lm.test_utility --compress $cpr --dataset mrpc --device cuda:0 --client_size 20 --top_k --k_ratio 0.1
#     done
# done

for k in ${k_ratios[@]}
do  
    echo "Train base model: Roberta, k ratio: $k"
    python3 -m lm.test_utility --dataset mrpc --device cuda:0 --client_size 20 --top_k --k_ratio $k
done