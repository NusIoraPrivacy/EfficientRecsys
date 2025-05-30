#!/bin/bash

# k_ratios=(0.05 0.1 0.15)
k_ratios=(0.1)

for k in ${k_ratios[@]}
do  
    echo "Test ResNet, Cifar 100, k ratio: $k"
    python -m randk.main --dataset cifar100 --device cuda:3 --k_sparse global_topk --k_ratio $k
done

# echo "Test ResNetReduce, Cifar 100"
# python -m randk.main --dataset cifar100 --model ResNet18Reduce --device cuda:3 --k_sparse none