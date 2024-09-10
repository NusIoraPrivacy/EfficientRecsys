import torch
from utils.globals import *
import numpy as np

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def str2type(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, torch.dtype):
        return v
    if "float32" in v.lower():
        return torch.float32
    elif "float16" in v.lower():
        return torch.float16

def cal_metrics(prediction, real_label, args):
    prediction = torch.tensor(prediction)
    real_label = torch.tensor(real_label)
    min_rate, max_rate = rating_range[args.dataset]
    prediction = torch.clip(prediction, min=min_rate, max=max_rate)
    mse = torch.mean((real_label - prediction)**2)
    rmse = torch.sqrt(torch.mean((real_label - prediction)**2))
    mae = torch.mean(torch.abs(real_label - prediction))
    return mse.item(), rmse.item(), mae.item()

def _evaluate_one_rating(predictions, rates, nagative_items, args):
    items = nagative_items + rates
    map_item_score = {}
    for item in items:
        map_item_score[item] = int(predictions[item])
    rank_list = heapq.nlargest(args.topk, map_item_score, key=map_item_score.get)
    rank_list  = set(rank_list)
    rates = set(rates)
    correct_pred = rates.intersection(rank_list)
    precision = len(correct_pred) / args.topk
    recall = len(correct_pred) / len(rates)
    return precision, recall

def get_noise_std(sensitivities, args):
    if isinstance(sensitivities, tuple):
        sensitivity = 0
        for s in sensitivities:
            sensitivity += s ** 2
        sensitivity = np.sqrt(sensitivity)
    else:
        sensitivity = sensitivities
    sigma = np.sqrt(2 * np.log(1.25 / args.delta)) * sensitivity / args.epsilon
    return sigma

def get_emb_size(base_model, args):
    private_params = private_param_dict[args.model]
    emb_dim = 0
    for name, param in base_model.named_parameters():
        if name in private_params:
            emb_dim += param.shape[-1]
    return emb_dim