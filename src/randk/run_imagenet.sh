#!/bin/bash

k_ratios=(0.05 0.1 0.15)

echo "Test ResNet Full, miniimagenet"
python -m randk.main --dataset miniimagenet --model ResNet18 --device cuda:3 --k_sparse none

for k in ${k_ratios[@]}
do  
    echo "Test ResNet, miniimagenet, k ratio: $k"
    python -m randk.main --dataset miniimagenet --device cuda:3 --k_sparse global_topk --k_ratio $k
done

