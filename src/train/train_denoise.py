import torch
from utils.globals import *
from utils.util import cal_metrics
import numpy as np
import heapq
from tqdm import tqdm
import os
import copy
import random
from utils.globals import private_param_dict

def test_model(model, user_id_list, item_id_list, test_data, args, denoise=False, denoise_model=None):
    prediction = []
    real_label = []
    sensitivity = norm_dict[args.model]
    # testing
    for i in range(len(user_id_list)):
        user_id_tensor = torch.tensor([i] * len(item_id_list)).to(args.device)
        item_id_tensor = torch.tensor(item_id_list).to(args.device)
        sigma = np.sqrt(2 * np.log(1.25 / args.delta)) * sensitivity / args.epsilon
        # p = model(user_id_tensor, item_id_tensor, noise_std=0)
        # print("Prediction before noise:", p)
        p, noise, init_user_emb = model(user_id_tensor, item_id_tensor, noise_std=sigma)
        # print("Prediction after noise:", p)
        if denoise:
            p = denoise_model(p, noise[0], init_user_emb[0])
        # obtain rmse
        ratings = test_data[i]
        real_label.extend([e[1] for e in ratings])
        prediction.extend([p[e[0]] for e in ratings])
    mse, rmse, mae = cal_metrics(prediction, real_label, args)
    return mse, rmse, mae

def train_demod(user_id_list, item_id_list, train_dataset, test_data, base_model, denoise_model, args):
    u_avg, u_std = distribution_dict[args.model]
    private_params = private_param_dict[args.model]
    emb_dim = 0
    for name, param in base_model.named_parameters():
        if name in private_params:
            emb_dim += param.shape[-1]

    sensitivity = norm_dict[args.model]
    item_size = len(item_id_list)
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.d_lr)
    best_rmse = 100
    best_model = copy.deepcopy(denoise_model)
    
    with tqdm(total=args.d_epochs) as pbar:
        for epoch in range(args.d_epochs):
            # generate synthetic user embedding
            user_id_list_shuffle = user_id_list.copy()
            random.shuffle(user_id_list_shuffle)
            this_user_list = user_id_list_shuffle[:args.d_batch_size]
            # generate noise, noise predictions, and clean predictions
            all_noise_predictions = torch.zeros(args.d_batch_size, item_size, device=args.device)
            all_noises = torch.zeros(args.d_batch_size, emb_dim, device=args.device)
            all_init_user_embs = torch.zeros(args.d_batch_size, emb_dim, device=args.device)
            all_clean_predictions = torch.zeros(args.d_batch_size, item_size, device=args.device)
            for i, user_idx in enumerate(this_user_list):
                noise_predictions, noises, init_user_embs, clean_predictions = train_dataset[user_idx]
                all_noise_predictions[i] = noise_predictions
                all_noises[i] = noises
                all_init_user_embs[i] = init_user_embs
                all_clean_predictions[i] = clean_predictions
            # use the generated data to train model
            denoise_ratings = denoise_model(all_noise_predictions, all_noises, all_init_user_embs)
            loss = denoise_model.get_loss(all_clean_predictions, denoise_ratings)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # test the performance of denoise model
            mse, rmse, mae = test_model(base_model, user_id_list, item_id_list, test_data, args, denoise=True, denoise_model=denoise_model)
            pbar.update(1)
            pbar.set_postfix(loss=loss.item(), rmse=rmse, mse=mse, mae=mae)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = copy.deepcopy(denoise_model)
    print("Best rmse:", best_rmse)
    save_dir = f"{args.root_path}/model/{args.dataset}/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(best_model.state_dict(), f"{save_dir}/denoise_best")